import sys

from ROOT import TFile,TStyle,TKey,TTree,TH1F,TH2D
from ROOT import TMath,TCanvas,TCut
from ROOT import gStyle,gROOT,gPad
from ROOT import kBlue

from cuts import *
from drawPlots import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)



if __name__ == "__main__":  

  inputFile = '/afs/cern.ch/user/d/dildick/work/GEM/CMSSW_6_2_0_pre5/src/gem_sh_ana.test.root'
  targetDir = './'
  
  ## extension for figures - add more?
  ext = ".png"
  
  ## GEM system settings
  nregion = 2
  nlayer = 2
  npart = 8
  
  ## Trees
  analyzer = "GEMSimHitAnalyzer"
  simHits = "GEMSimHits"
  simTracks = "Tracks"

  ## muon selection
  muonSelection = [TCut("TMath::Abs(particleType)==13"),TCut("TMath::Abs(particleType)!=13"),TCut("TMath::Abs(particleType)==13 && TMath::Abs(particleType)!=13")]
  titlePrefix = ["Muon","Non muon","All"]
  histSuffix = ["_muon","_nonmuon","_all"]

  ## Style
  gStyle.SetStatStyle(0);

  ## input
  file = TFile.Open(inputFile)
  if not file:
    sys.exit('Input ROOT file %s is missing.' %(inputFile))

  dirAna = file.Get(analyzer)
  if not dirAna:
    sys.exit('Directory %s does not exist.' %(dirAna))
    
  treeHits = dirAna.Get(simHits)
  if not treeHits:
    sys.exit('Tree %s does not exist.' %(treeHits))
  
  for i in range(len(muonSelection)):

    sel = muonSelection[i]
    pre = titlePrefix[i]
    suff = histSuffix[i]
 
    draw_occ(targetDir, "sh_xy_rm1_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]", 
 	     "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s && %s" %(rm1.GetTitle(), l1.GetTitle(), sel.GetTitle())), "COLZ")
    draw_occ(targetDir, "sh_xy_rm1_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]", 
 	     "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s && %s" %(rm1.GetTitle(), l2.GetTitle(), sel.GetTitle())), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_l1" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]", 
 	     "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s && %s" %(rp1.GetTitle(), l1.GetTitle(), sel.GetTitle())), "COLZ")
    draw_occ(targetDir, "sh_xy_rp1_l2" + suff, ext, treeHits, pre + " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]", 
 	     "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", TCut("%s && %s && %s" %(rp1.GetTitle(), l2.GetTitle(), sel.GetTitle())), "COLZ")
    
    draw_occ(targetDir, "sh_zr_rm1" + suff, ext, treeHits, pre + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]", 
             "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", TCut('%s && %s'%(rm1.GetTitle(), sel.GetTitle())), "COLZ")
    draw_occ(targetDir, "sh_zr_rp1" + suff, ext, treeHits, pre + " SimHit occupancy: region1;globalZ [cm];globalR [cm]", 
 	     "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", TCut('%s && %s'%(rp1.GetTitle(), sel.GetTitle())), "COLZ")
    
    draw_1D(targetDir, "sh_tof_rm1_l1" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer1;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", TCut("%s && %s && %s" %(rm1.GetTitle(), l1.GetTitle(), sel.GetTitle())))
    draw_1D(targetDir, "sh_tof_rm1_l2" + suff, ext, treeHits, pre + " SimHit TOF: region-1, layer2;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", TCut("%s && %s && %s" %(rm1.GetTitle(), l2.GetTitle(), sel.GetTitle())))
    draw_1D(targetDir, "sh_tof_rp1_l1" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer1;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", TCut("%s && %s && %s" %(rp1.GetTitle(), l1.GetTitle(), sel.GetTitle())))
    draw_1D(targetDir, "sh_tof_rp1_l2" + suff, ext, treeHits, pre + " SimHit TOF: region1, layer2;Time of flight [ns];entries", 
            "h_", "(40,18,22)", "timeOfFlight", TCut("%s && %s && %s" %(rp1.GetTitle(), l2.GetTitle(), sel.GetTitle())))
    
    ## momentum plot
    c = TCanvas("c","c",600,600)
    c.Clear()
    treeHits.Draw("pabs>>hh(200,0.,200.)",sel)
    h = TH1F(gDirectory.Get("hh"))
    gPad.SetLogx(0)
    gPad.SetLogy(1)
    h.SetTitle(pre + " SimHits absolute momentum;Momentum [GeV/c];entries")       
    h.SetLineWidth(2)
    h.SetLineColor(kBlue)
    h.Draw("")        
    c.SaveAs(targetDir +"sh_momentum" + suff + ext)
    
    
    draw_1D(targetDir, "sh_pdgid" + suff, ext, treeHits, pre + " SimHit PDG Id;PDG Id;entries", 
   	    "h_", "(200,-100.,100.)", "particleType", sel)
    
##     /// eta occupancy plot
##     int region=0;
##     int layer=0;
##     int roll=0;  
##     int particletype=0;
##     TBranch *b_region;
##     TBranch *b_layer;
##     TBranch *b_roll;
##     TBranch *b_particleType;  
##     treeHits->SetBranchAddress("region", &region, &b_region)
##     treeHits->SetBranchAddress("layer", &layer, &b_layer)
##     treeHits->SetBranchAddress("roll", &roll, &b_roll)
##     treeHits->SetBranchAddress("particleType", &particletype, &b_particleType)
##     h = new TH1D("h", pre + " SimHit occupancy in eta partitions; occupancy in #eta partition; entries",4*npart_,1.,1.+4*npart_)
##     int nbytes = 0;
##     int nb = 0;
##     for (Long64_t jentry=0; jentry<treeHits->GetEntriesFast()jentry++) {
##       Long64_t ientry = treeHits->LoadTree(jentry)
##       if (ientry < 0) break;
##       nb = treeHits->GetEntry(jentry)   
##       nbytes += nb;
##       switch(uSel){
##       case 0:
## 	if (abs(particletype)==13) h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) )
## 	break;
##       case 1:
## 	if (abs(particletype)!=13) h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) )
## 	break;
##       case 2:
## 	h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) )
## 	break;
##       }
##     }    
##     c->Clear()  
##     gPad->SetLogx(0)
##     gPad->SetLogy(0)
##     int ibin(1)
##     for (int iregion = 1; iregion<nregion_+1; ++iregion){
##       TString region( (iregion == 1) ? "-" : "+" )
##       for (int ilayer = 1; ilayer<nregion_+1; ++ilayer){
## 	TString layer( TString::Itoa(ilayer,10)) 
## 	for (int ipart = 1; ipart<npart_+1; ++ipart){
## 	  TString part( TString::Itoa(ipart,10)) 
## 	  h->GetXaxis()->SetBinLabel(ibin,region+layer+part)
## 	  ++ibin;
## 	}
##       }
##     }
    
##     h->SetMinimum(0.)
##     h->SetLineWidth(2)
##     h->SetLineColor(kBlue)
##     h->Draw("")        
##     c->SaveAs(targetDir +"sh_globalEta" + suff + ext)
    
##     /// energy loss plot
##     h = new TH1D("h","",60,0.,6000.)
##     Float_t energyLoss=0;
##     TBranch *b_energyLoss;
##     treeHits->SetBranchAddress("energyLoss", &energyLoss, &b_energyLoss)
##     for (Long64_t jentry=0; jentry<treeHits->GetEntriesFast()jentry++) {
##       Long64_t ientry = treeHits->LoadTree(jentry)
##       if (ientry < 0) break;
##       nb = treeHits->GetEntry(jentry)   
##       nbytes += nb;
##       switch(uSel){
##       case 0:
## 	if (abs(particletype)==13) h->Fill( energyLoss*1.e9 )
## 	break;
##       case 1:
## 	if (abs(particletype)!=13) h->Fill( energyLoss*1.e9 )
##       break;
##       case 2:
## 	h->Fill( energyLoss*1.e9 )
## 	break;
##       }
##     }
##     c->Clear()  
##     gPad->SetLogx(0)
##     gPad->SetLogy(0)
##     h->SetTitle(pre + " SimHit energy loss;Energy loss [eV];entries")
##     h->SetMinimum(0.)
##     h->SetLineWidth(2)
##     h->SetLineColor(kBlue)
##     h->Draw("")        
##     c->SaveAs(targetDir + "sh_energyloss" + suff + ext)

##   }
  
  treeTracks = dirAna.Get(simTracks)
  if not treeTracks:
    sys.exit('Tree %s does not exist.' %(treeTracks))

  draw_geff(targetDir, "eff_eta_track_sh_gem_l1or2", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s || %s" %(ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_sh_gem_l1", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_sh_gem_l2", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), ok_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_eta_track_sh_gem_l1and2", ext, treeTracks, 
            "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack |#eta|;Eff.", 
            "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", TCut(""), TCut("%s && %s" %(ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())), "P", kBlue)
    
  draw_geff(targetDir, "eff_phi_track_sh_gem_l1or2", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, TCut("%s || %s" %(ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())), "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_sh_gem_l1", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, ok_gL1sh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_sh_gem_l2", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, ok_gL2sh, "P", kBlue)
  draw_geff(targetDir, "eff_phi_track_sh_gem_l1and2", ext, treeTracks, 
  	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack #phi [rad];Eff.", 
  	    "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", ok_eta, TCut("%s && %s" %(ok_gL1sh.GetTitle(),ok_gL2sh.GetTitle())), "P", kBlue)

## //   draw_geff(targetDir, "eff_globalx_track_sh_gem_l1or2", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack globalx;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globalX", "", ok_gL1sh || ok_gL2sh, "P", kBlue)
## //   draw_geff(targetDir, "eff_globalx_track_sh_gem_l1", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack globalx;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globalX", "", ok_gL1sh, "P", kBlue)
## //   draw_geff(targetDir, "eff_globalx_track_sh_gem_l2", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack globalx;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globalx_layer1", "", ok_gL2sh, "P", kBlue)
## //   draw_geff(targetDir, "eff_globalx_track_sh_gem_l1and2", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack globalx;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globalx_layer1", "", ok_gL1sh && ok_gL2sh, "P", kBlue)

## //   draw_geff(targetDir, "eff_globaly_track_sh_gem_l1or2", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 or GEMl2;SimTrack globaly;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globaly_layer1", "", ok_gL1sh || ok_gL2sh, "P", kBlue)
## //   draw_geff(targetDir, "eff_globaly_track_sh_gem_l1", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1;SimTrack globaly;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globaly_layer1", "", ok_gL1sh, "P", kBlue)
## //   draw_geff(targetDir, "eff_globaly_track_sh_gem_l2", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl2;SimTrack globaly;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globaly_layer1", "", ok_gL2sh, "P", kBlue)
## //   draw_geff(targetDir, "eff_globaly_track_sh_gem_l1and2", ext, treeTracks, 
## // 	    "Eff. for a SimTrack to have an associated GEM SimHit in GEMl1 and GEMl2;SimTrack globaly;Eff.", 
## // 	    "h_", "(250,-250.,250)", "globaly_layer1", "", ok_gL1sh && ok_gL2sh, "P", kBlue)

  draw_1D(targetDir, "track_pt", ext, treeTracks, "Track p_{T};Track p_{T} [GeV];Entries", "h_", "(100,0,200)", "pt", "")
  draw_1D(targetDir, "track_eta", ext, treeTracks, "Track |#eta|;Track |#eta|;Entries", "h_", "(100,1.5,2.2)", "eta", "")
  draw_1D(targetDir, "track_phi", ext, treeTracks, "Track #phi;Track #phi [rad];Entries", "h_", "(100,-3.14159265358979312,3.14159265358979312)", "phi", "")
  
