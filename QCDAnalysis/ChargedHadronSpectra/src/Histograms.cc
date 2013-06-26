#include "QCDAnalysis/ChargedHadronSpectra/interface/Histograms.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"

#include <iostream>

#include <cmath>

/*****************************************************************************/
#define nCharges 2
enum Charges
{ pos, neg, zero, undefined  };

const char *chargeName[nCharges + 2] =
{ "pos", "neg", "zero", "undefined" };

/*****************************************************************************/
#define nParticles 21
enum Particles
{
  pip, pim, kap, kam,
  prp, prm, elp, elm,
  hap, ham,
  gam, k0s, lam, ala,
  rho, kst, aks, phi,
  sip, asi,
  any
};

const char *partName[nParticles] =
{
  "pip", "pim", "kap", "kam",
  "prp", "prm", "elp", "elm",
  "hap", "ham",
  "gam", "k0s", "lam", "ala",
  "rho", "kst", "aks", "phi",
  "sip", "asi",
  "any"
};

const int partCharge[nParticles] =
{
  pos , neg , pos , neg ,
  pos , neg , pos , neg ,
  pos , neg ,
  zero, zero, zero, zero,
  zero, zero, zero, zero,
  pos , neg ,
  undefined
};

/*****************************************************************************/
#define nFeedDowns 18
const std::pair<int,int> feedDown[nFeedDowns] =
{
  std::pair<int,int>(k0s, pip), std::pair<int,int>(k0s, pim),
  std::pair<int,int>(lam, prp), std::pair<int,int>(lam, pim),
  std::pair<int,int>(ala, prm), std::pair<int,int>(ala, pip),
  std::pair<int,int>(sip, prp), std::pair<int,int>(asi, prm),
  std::pair<int,int>(any, pip), std::pair<int,int>(any, pim),
  std::pair<int,int>(any, kap), std::pair<int,int>(any, kam),
  std::pair<int,int>(any, prp), std::pair<int,int>(any, prm),
  std::pair<int,int>(any, elp), std::pair<int,int>(any, elm),
  std::pair<int,int>(any, hap), std::pair<int,int>(any, ham)
};

#define nResonances 4
const std::pair<int,int> resonance[nResonances] =
{
 std::pair<int,int>(pip, pim),
 std::pair<int,int>(kap, pim),
 std::pair<int,int>(pip, kam),
 std::pair<int,int>(kap, kam)
};

/*****************************************************************************/
Histograms::Histograms(const edm::ParameterSet& pset)
{
  fillHistograms         = pset.getParameter<bool>("fillHistograms");
  fillNtuples            = pset.getParameter<bool>("fillNtuples");

  std::string histoFileLabel = pset.getParameter<std::string>("histoFile");
  histoFile = new TFile(histoFileLabel.c_str(),"recreate");

  std::string ntupleFileLabel = pset.getParameter<std::string>("ntupleFile");
  ntupleFile = new TFile(ntupleFileLabel.c_str(),"recreate");

//  resultFile->cd();
}

/*****************************************************************************/
Histograms::~Histograms()
{
}

/****************************************************************************/
int Histograms::getParticle(int id)
{
  switch(id)
  {
    case   211 : return pip; break;
    case  -211 : return pim; break;

    case   321 : return kap; break;
    case  -321 : return kam; break;

    case  2212 : return prp; break;
    case -2212 : return prm; break;

    case    11 : return elp; break;
    case   -11 : return elm; break;

    //  SimG4Core/Notification/src/G4TrackToParticleID.cc
    // "deuteron" = -100;  1p 1n
    // "triton"   = -101;  1p 2n
    // "alpha"    = -102;  2p 2n
    // "He3"      = -104;  2p 1n

    case    22 : return gam; break;
    case   310 : return k0s; break;
    case  3122 : return lam; break;
    case -3122 : return ala; break;

/*
    case   113 : return rho; break;
    case   313 : return kst; break;
    case  -313 : return aks; break;
    case   333 : return phi; break;
*/

    case  3222 : return sip; break;
    case -3222 : return asi; break;

    default    : return -1;  break;
  }
}

/****************************************************************************/
int Histograms::getCharge(int charge)
{ 
  if(charge > 0) return pos;
            else return neg;
}

/*****************************************************************************/
void Histograms::declareHistograms()
{
  if(fillNtuples)
  {
    TString leafStr;

    trackTrees.push_back(new TTree("simTrackTree","simTrackTree"));
    leafStr = "ids/I:etas/F:pts/F:acc/I:prim/I:nrec/I:ntrkr/I";
    trackTrees[0]->Branch("simTrackValues", &simTrackValues, leafStr.Data());

    trackTrees.push_back(new TTree("recTrackTree","recTrackTree"));
    leafStr = "charge/I:etar/F:ptr/F:phir/F:zr/F:logpr/F:logde/F:nhitr/I:prim/I:nsim/I:ids/I:parids/I:etas/F:pts/F:ntrkr/I";
    trackTrees[1]->Branch("recTrackValues", &recTrackValues, leafStr.Data());

    trackTrees.push_back(new TTree("recVzeroTree","recVzeroTree"));
    leafStr = "etar/F:ptr/F:ima/F:rhor/F";
    trackTrees[2]->Branch("recVzeroValues", &recVzeroValues, leafStr.Data());

    trackTrees.push_back(new TTree("eventInfoTree","eventInfoTree"));
    leafStr = "proc/I:strk/I:ntrkr/I";
    trackTrees[3]->Branch("eventInfoValues", &eventInfoValues, leafStr.Data());
  }

  if(fillHistograms)
  {
  /////////////////////////////
  // Pt
  const double small = 1e-3;
  double pt;

  for(pt =  0; pt <  1 - small; pt += 0.05) ptBins.push_back(pt);
  for(pt =  1; pt <  2 - small; pt += 0.1 ) ptBins.push_back(pt);
  for(pt =  2; pt <  4 - small; pt += 0.2 ) ptBins.push_back(pt);
  for(pt =  4; pt <  8 - small; pt += 0.5 ) ptBins.push_back(pt);
  for(pt =  8; pt < 16 - small; pt += 1.  ) ptBins.push_back(pt);
  for(pt = 16; pt < 32 - small; pt += 2.  ) ptBins.push_back(pt);
  for(pt = 32; pt < 64 - small; pt += 4.  ) ptBins.push_back(pt);

  constexpr float ratMin   = 0.5;
  constexpr float ratMax   = 1.5;
  constexpr float ratWidth = 1./200;

  for(double rat = ratMin; rat < ratMax + ratWidth/2; rat += ratWidth)
    ratBins.push_back(rat);

  /////////////////////////////
  // Eta (-3,3)
  constexpr float etaMin   = -3.0;
  constexpr float etaMax   =  3.0;
  constexpr float etaWidth =  0.2;

  for(double eta = etaMin; eta < etaMax + etaWidth/2; eta += etaWidth)
    etaBins.push_back(eta);

//  for(double eta = etaMin; eta < etaMax + etaWidth/2; eta += etaWidth/10)
  for(double eta = etaMin; eta < etaMax + etaWidth/2; eta += etaWidth/5)
    metaBins.push_back(eta);

  constexpr float zMin   = -20.;
  constexpr float zMax   =  20.;
//  constexpr float zWidth =  0.1;
  constexpr float zWidth =  0.2;

  for(double z = zMin; z < zMax + zWidth/2; z += zWidth)
    zBins.push_back(z);

  /////////////////////////////
  // Number of recontructed tracks
  constexpr float ntrkMin   =  0.5;
// FIXME
//  constexpr float ntrkMax   = 200.;
//  constexpr float ntrkWidth =   5.;
  constexpr float ntrkMax   = 1000.;
  constexpr float ntrkWidth =   10.;
  
  for(double ntrk = ntrkMin; ntrk < ntrkMax + ntrkWidth; ntrk += ntrkWidth)
    ntrkBins.push_back(ntrk);


  char histName[256];

  ///////////////////
  // EventInfo
  sprintf(histName,"heve");
  heve.push_back(new TH1F(histName,histName, 200, -0.5,199.5));

  sprintf(histName,"hsdx");
  heve.push_back(new TH1F(histName,histName, 200, -0.5,199.5));

  sprintf(histName,"hddx");
  heve.push_back(new TH1F(histName,histName, 200, -0.5,199.5));

  sprintf(histName,"hndx");
  heve.push_back(new TH1F(histName,histName, 200, -0.5,199.5));

  sprintf(histName,"hder");
  hder.push_back(new TH2F(histName,histName, 200, -0.5,199.5,
                                             200, -0.5,199.5));

  ///////////////////
  // SimTrack
  for(int part = pip; part <= ala; part++)
  {
    // simulated
    sprintf(histName,"hsim_%s", partName[part]);
    hsim.push_back(new TH3F(histName,histName,
                            etaBins.size()-1,  &etaBins[0],
                             ptBins.size()-1,   &ptBins[0],
                           ntrkBins.size()-1, &ntrkBins[0]));
   
    // accepted 
    sprintf(histName,"hacc_%s", partName[part]);
    hacc.push_back(new TH3F(histName,histName,
                            etaBins.size()-1,  &etaBins[0],
                             ptBins.size()-1,   &ptBins[0],
                           ntrkBins.size()-1, &ntrkBins[0]));
  
    // reconstructed/efficiency
    sprintf(histName,"href_%s", partName[part]);
    href.push_back(new TH3F(histName,histName,
                            etaBins.size()-1,  &etaBins[0],
                             ptBins.size()-1,   &ptBins[0],
                           ntrkBins.size()-1, &ntrkBins[0]));
  
    // multiply reconstructed
    sprintf(histName,"hmul_%s", partName[part]);
    hmul.push_back(new TH3F(histName,histName,
                            etaBins.size()-1,  &etaBins[0],
                             ptBins.size()-1,   &ptBins[0],
                           ntrkBins.size()-1, &ntrkBins[0]));
  }

  ///////////////////
  // RecTrack
  for(int charge = 0; charge < nCharges; charge++)
  {
    sprintf(histName,"hall_%s",chargeName[charge]);
    hall.push_back(new TH3F(histName,histName,
                            etaBins.size()-1,  &etaBins[0],
                             ptBins.size()-1,   &ptBins[0],
                           ntrkBins.size()-1, &ntrkBins[0]));

    sprintf(histName,"hdac_%s",chargeName[charge]);
    hdac.push_back(new TH2F(histName,histName, 
                           metaBins.size()-1, &metaBins[0],
                              zBins.size()-1,    &zBins[0]));

    ///////////////////
    // RecTrack -- FakeRate
    sprintf(histName,"hfak_%s",chargeName[charge]);
    hfak.push_back(new TH3F(histName,histName,
                            etaBins.size()-1,  &etaBins[0],
                             ptBins.size()-1,   &ptBins[0],
                           ntrkBins.size()-1, &ntrkBins[0]));
  }

  ///////////////////
  // RecTrack -- Resolution, bias
  for(int part = pip; part <= ala; part++)
  {
    // value
    sprintf(histName,"hvpt_%s",partName[part]);
    hvpt.push_back(new TH3F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0],
                             ptBins.size()-1,  &ptBins[0]));

    // ratio
    sprintf(histName,"hrpt_%s",partName[part]);
    hrpt.push_back(new TH3F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0],
                            ratBins.size()-1, &ratBins[0]));

    sprintf(histName,"hsp0_%s",partName[part]);
    hsp0.push_back(new TH2F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0]));

    sprintf(histName,"hsp1_%s",partName[part]);
    hsp1.push_back(new TH2F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0]));

    sprintf(histName,"hsp2_%s",partName[part]);
    hsp2.push_back(new TH2F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0]));
  }

  ///////////////////
  // RecTrack -- FeedDown
  for(int k = 0; k < nFeedDowns; k++)
  {
    sprintf(histName,"hpro_%s_%s", partName[feedDown[k].first], // produced
                                   partName[feedDown[k].second]);
    hpro.push_back(new TH2F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0]));

    sprintf(histName,"hdec_%s_%s", partName[feedDown[k].first], // decay
                                   partName[feedDown[k].second]);
    hdec.push_back(new TH2F(histName,histName,
                            etaBins.size()-1, &etaBins[0],
                             ptBins.size()-1,  &ptBins[0]));
  }

  ///////////////////
  // EnergyLoss
  constexpr float lpMin   = -3; // 50 MeV
  constexpr float lpMax   =  2; // 7.4 GeV
  constexpr float lpWidth = (lpMax - lpMin)/100;
  for(double lp = lpMin; lp < lpMax + lpWidth/2; lp += lpWidth)
    lpBins.push_back(lp);

  const float ldeMin   = log(1);
  const float ldeMax   = log(100);
  const float ldeWidth = (ldeMax - ldeMin)/250;
  for(double lde = ldeMin; lde < ldeMax + ldeWidth/2; lde += ldeWidth)
    ldeBins.push_back(lde);

  for(double nhit = -0.5; nhit < 50; nhit += 1)
    nhitBins.push_back(nhit);

  for(int charge = 0; charge < nCharges; charge++)
  {
    // All hits
    // dE/dx
    sprintf(histName,"helo_%s", chargeName[charge]);
    helo.push_back(new TH3F(histName,histName,
                           etaBins.size()-1, &etaBins[0],
                            ptBins.size()-1,  &ptBins[0],
                           ldeBins.size()-1, &ldeBins[0]));

    // Number of hits used
    sprintf(histName,"hnhi_%s", chargeName[charge]);
    hnhi.push_back(new TH3F(histName,histName,
                           etaBins.size()-1,  &etaBins[0],
                            ptBins.size()-1,   &ptBins[0],
                          nhitBins.size()-1, &nhitBins[0]));

    // Demo plot
    sprintf(histName,"held_%s", chargeName[charge]);
    held.push_back(new TH2F(histName,histName,
                            lpBins.size()-1,  &lpBins[0],
                           ldeBins.size()-1, &ldeBins[0])); 
  }

/*
  for(int charge = 0; charge < nCharges; charge++)
  {
    // Strip hits
    // dE/dx
    sprintf(histName,"selo_%s", chargeName[charge]);
    selo.push_back(new TH3F(histName,histName,
                           etaBins.size()-1, &etaBins[0],
                            ptBins.size()-1,  &ptBins[0],
                           ldeBins.size()-1, &ldeBins[0]));

    // Number of hits used
    sprintf(histName,"snhi_%s", chargeName[charge]);
    snhi.push_back(new TH3F(histName,histName,
                           etaBins.size()-1,  &etaBins[0],
                            ptBins.size()-1,   &ptBins[0],
                          nhitBins.size()-1, &nhitBins[0]));
    
    // Demo plot
    sprintf(histName,"seld_%s", chargeName[charge]);
    seld.push_back(new TH2F(histName,histName,
                            lpBins.size()-1,  &lpBins[0],
                           ldeBins.size()-1, &ldeBins[0]));
  }
*/

  ///////////////////
  // Invariant mass
  constexpr float rhoMin   = 0.;
  constexpr float rhoMax   = 5.;
  constexpr float rhoWidth = 0.2; 
  for(double rho_ = rhoMin; rho_ < rhoMax + rhoWidth/2; rho_ += rhoWidth)
    rhoBins.push_back(rho_);


  for(int part = gam; part <= phi; part++)
  {
    float imMin = 0;
    float imMax = 0;
    float imWidth = 0;

    if(part == gam) { imMin = 0.0; imMax = 0.2; imWidth = 0.005; }
    if(part == k0s) { imMin = 0.3; imMax = 0.7; imWidth = 0.005; }
    if(part == lam ||
       part == ala) { imMin = 1.0; imMax = 1.3; imWidth = 0.002; }

    if(part == rho) { imMin = 0.2; imMax = 1.2; imWidth = 0.010; }
    if(part == kst ||
       part == aks) { imMin = 0.6; imMax = 1.6; imWidth = 0.010; }
    if(part == phi) { imMin = 0.9; imMax = 1.1; imWidth = 0.002; }

    std::vector<double> imBins;
    double im;
    for(im = imMin; im < imMax + imWidth/2; im += imWidth)
      imBins.push_back(im);

    if(imWidth > 0)
    {
      sprintf(histName,"hima_%s", partName[part]);
      hima.push_back(new TH3F(histName,histName,
                              etaBins.size()-1, &etaBins[0],
                               ptBins.size()-1,  &ptBins[0],
                               imBins.size()-1,  &imBins[0]));

      if(part >= rho && part <= phi)
      {
      sprintf(histName,"himp_%s", partName[part]);
      hima.push_back(new TH3F(histName,histName,
                              etaBins.size()-1, &etaBins[0],
                               ptBins.size()-1,  &ptBins[0],
                               imBins.size()-1,  &imBins[0]));

      sprintf(histName,"himm_%s", partName[part]);
      hima.push_back(new TH3F(histName,histName,
                              etaBins.size()-1, &etaBins[0],
                               ptBins.size()-1,  &ptBins[0],
                               imBins.size()-1,  &imBins[0]));

      sprintf(histName,"himx_%s", partName[part]);
      hima.push_back(new TH3F(histName,histName,
                              etaBins.size()-1, &etaBins[0],
                               ptBins.size()-1,  &ptBins[0],
                               imBins.size()-1,  &imBins[0]));
      }

      sprintf(histName,"hrho_%s", partName[part]);
      hrho.push_back(new TH3F(histName,histName,
                              rhoBins.size()-1, &rhoBins[0],
                               ptBins.size()-1,  &ptBins[0],
                               imBins.size()-1,  &imBins[0]));
    }
  }
  }
}

/****************************************************************************/
void Histograms::fillEventInfo(int proc, int strk, int ntrk)
{
  if(fillNtuples)
  {
    EventInfo_t e;
    e.proc  = proc;
    e.strk  = strk;
    e.ntrkr = ntrk;

    eventInfoValues = e;

    trackTrees[3]->Fill();
  }
  
  if(fillHistograms)
  {
  heve[0]->Fill(ntrk);

  if(proc == 92 || proc == 93)
    heve[1]->Fill(ntrk); // hsdx

  if(proc == 94)
    heve[2]->Fill(ntrk); // hddx

  if(!(proc == 92 || proc == 93 || proc == 94))
    heve[3]->Fill(ntrk); // hndx

  // For multiplicity, detector response matrix
  hder[0]->Fill(strk,ntrk);
  }
}

/****************************************************************************/
void Histograms::fillSimHistograms(const SimTrack_t & s)
{
  if(fillNtuples)
  { 
    if(s.prim)
    {
      simTrackValues = s;
      trackTrees[0]->Fill();
    }
  }

  if(fillHistograms)
  {
    int part = getParticle(s.ids);   

    if(pip <= part && part <= ala && s.prim)
    {
                     hsim[part]->Fill(s.etas, s.pts, s.ntrkr);
      if(s.acc)      hacc[part]->Fill(s.etas, s.pts, s.ntrkr);
if(s.acc)
{
      if(s.nrec > 0) href[part]->Fill(s.etas, s.pts, s.ntrkr);
      if(s.nrec > 1) hmul[part]->Fill(s.etas, s.pts, s.ntrkr);
}
  
      if(partCharge[part] == pos || partCharge[part] == neg)
      {
        if(partCharge[part] == pos) part = hap;
        if(partCharge[part] == neg) part = ham;
  
                       hsim[part]->Fill(s.etas, s.pts, s.ntrkr);
        if(s.acc)      hacc[part]->Fill(s.etas, s.pts, s.ntrkr);
if(s.acc)
{
        if(s.nrec > 0) href[part]->Fill(s.etas, s.pts, s.ntrkr);
        if(s.nrec > 1) hmul[part]->Fill(s.etas, s.pts, s.ntrkr);
}
      }
    }
  }
}

/****************************************************************************/
void Histograms::fillRecHistograms(const RecTrack_t & r)
{
  if(fillNtuples)
  {
    if(r.prim)
    {
      recTrackValues = r;
      trackTrees[1]->Fill();
    }
  }

  if(fillHistograms)
  {
  int charge = getCharge(r.charge);
  double p = exp(r.logpr);

  if(r.prim)
  {
    hall[charge]->Fill(r.etar, r.ptr, r.ntrkr);

    if(r.ptr > 0.3) // !!!
    hdac[charge]->Fill(r.etar, r.zr);

    if(r.nsim == 0)
    hfak[charge]->Fill(r.etar, r.ptr, r.ntrkr);
    
    if(r.nsim == 1)
    {
      int part = getParticle(r.ids);
      int moth = getParticle(r.parids);

      if(pip <= part && part <= ala)
      {
        hvpt[part]->Fill(r.etas, r.pts, r.ptr      ); // value
        hrpt[part]->Fill(r.etas, r.pts, r.ptr/r.pts); // ratio 

        hsp0[part]->Fill(r.etas, r.pts);      // sum p^0
        hsp1[part]->Fill(r.etas, r.pts, p);   // sum p^1
        hsp2[part]->Fill(r.etas, r.pts, p*p); // sum p^2
      }

      for(int k = 0; k < nFeedDowns; k++)
        if(part == feedDown[k].second) // daughter same
        {
          hpro[k]->Fill(r.etar, r.ptr);

          if((r.parids != 0 && feedDown[k].first == any) ||
                               feedDown[k].first == moth) 
          hdec[k]->Fill(r.etar, r.ptr);
        }

      if(partCharge[part] == pos || partCharge[part] == neg)
      {
        if(partCharge[part] == pos) part = hap;
        if(partCharge[part] == neg) part = ham;

        hvpt[part]->Fill(r.etas, r.pts, r.ptr      ); // value
        hrpt[part]->Fill(r.etas, r.pts, r.ptr/r.pts); // ratio

        hsp0[part]->Fill(r.etas, r.pts);      // sum p^0
        hsp1[part]->Fill(r.etas, r.pts, p);   // sum p^1
        hsp2[part]->Fill(r.etas, r.pts, p*p); // sum p^2

        for(int k = 0; k < nFeedDowns; k++)
        if(part == feedDown[k].second) // daughter same
        {
          hpro[k]->Fill(r.etar, r.ptr);

          if((r.parids != 0 && feedDown[k].first == any) ||
                               feedDown[k].first == moth)
          hdec[k]->Fill(r.etar, r.ptr);
        }
      }
    }

    // All hits
    helo[charge]->Fill(r.etar, r.ptr, r.logde);
    hnhi[charge]->Fill(r.etar, r.ptr, r.nhitr);
    held[charge]->Fill(r.logpr, r.logde);

    // Strip hits
/*
    selo[charge]->Fill(r.etar, r.ptr, r.logde_strip);
    snhi[charge]->Fill(r.etar, r.ptr, r.nhitr_strip);
    seld[charge]->Fill(r.logpr, r.logde_strip);
*/
  }
  }
}

/****************************************************************************/
void Histograms::fillVzeroHistograms(const RecVzero_t & v, int part)
{
  if(fillNtuples)
  {
    recVzeroValues = v;
    trackTrees[2]->Fill();
  }

  if(fillHistograms)
    hima[part]->Fill(v.etar, v.ptr, v.ima);
}

/****************************************************************************/
void Histograms::writeHistograms()
{
  typedef std::vector<TH3F *>::const_iterator H3I;
  typedef std::vector<TH2F *>::const_iterator H2I;
  typedef std::vector<TH1F *>::const_iterator H1I;
  typedef std::vector<TTree *>::const_iterator TI;

  if(fillHistograms)
  {
  histoFile->cd();
  for(H1I h = heve.begin(); h!= heve.end(); h++) (*h)->Write();
  for(H2I h = hder.begin(); h!= hder.end(); h++) (*h)->Write();

  for(H3I h = hsim.begin(); h!= hsim.end(); h++) (*h)->Write();
  for(H3I h = hacc.begin(); h!= hacc.end(); h++) (*h)->Write();
  for(H3I h = href.begin(); h!= href.end(); h++) (*h)->Write();
  for(H3I h = hmul.begin(); h!= hmul.end(); h++) (*h)->Write();

  for(H3I h = hall.begin(); h!= hall.end(); h++) (*h)->Write();
  for(H2I h = hdac.begin(); h!= hdac.end(); h++) (*h)->Write();

  for(H3I h = hvpt.begin(); h!= hvpt.end(); h++) (*h)->Write();
  for(H3I h = hrpt.begin(); h!= hrpt.end(); h++) (*h)->Write();

  for(H2I h = hsp0.begin(); h!= hsp0.end(); h++) (*h)->Write();
  for(H2I h = hsp1.begin(); h!= hsp1.end(); h++) (*h)->Write();
  for(H2I h = hsp2.begin(); h!= hsp2.end(); h++) (*h)->Write();

  for(H3I h = hfak.begin(); h!= hfak.end(); h++) (*h)->Write();

  for(H2I h = hpro.begin(); h!= hpro.end(); h++) (*h)->Write();
  for(H2I h = hdec.begin(); h!= hdec.end(); h++) (*h)->Write();

  for(H3I h = helo.begin(); h!= helo.end(); h++) (*h)->Write();
  for(H3I h = hnhi.begin(); h!= hnhi.end(); h++) (*h)->Write();
  for(H2I h = held.begin(); h!= held.end(); h++) (*h)->Write();

  for(H3I h = hima.begin(); h!= hima.end(); h++) (*h)->Write();
  for(H3I h = hrho.begin(); h!= hrho.end(); h++) (*h)->Write();
  histoFile->Close();
  }

  if(fillNtuples)
  {
  ntupleFile->cd();
  for(TI t = trackTrees.begin(); t!= trackTrees.end(); t++) (*t)->Write();
  ntupleFile->Close();
  }
}

