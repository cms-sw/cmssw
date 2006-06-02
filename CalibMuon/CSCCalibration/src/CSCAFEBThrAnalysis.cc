#include "CalibMuon/CSCCalibration/interface/CSCAFEBThrAnalysis.h"
#include "CalibMuon/CSCCalibration/interface/CSCToAFEB.h"
#include <CalibMuon/CSCCalibration/interface/CSCFitAFEBThr.h>
#include "CLHEP/Random/RandGaussQ.h"
#include "TMath.h"

class CSCFitAFEBThr;

CSCAFEBThrAnalysis::CSCAFEBThrAnalysis() {

hist_file=0; // set to null

nmbev=0;
nmbev_no_wire=0;

npulses=10;
vecDac.clear();
BegDac=1; 
EvDac=10; 
StepDac=1;

m_wire_dac.clear();
m_res_for_db.clear();
mh_ChanEff.clear();
mh_FirstTime.clear();
mh_AfebDac.clear();

}

void CSCAFEBThrAnalysis::setup(const std::string& histoFileName) {
  /// open the histogram file
  hist_file=new TFile(histoFileName.c_str(),"RECREATE");
  hist_file->cd();
}
 
void CSCAFEBThrAnalysis::bookForId(int flag, const int& idint,
                                             const std::string& idstring ) {
  hist_file->cd();

  std::ostringstream ss;

  if(flag==100) {
  ss <<idint<<"_Anode_First_Time";
  mh_FirstTime[idint]=new TH2F(ss.str().c_str(),"",675,0.0,675.0,50,0.0,10.0);
  mh_FirstTime[idint]->GetXaxis()->SetTitle("(AFEB-1)*16+ch");
  mh_FirstTime[idint]->GetYaxis()->SetTitle("Anode First Time Bin");
  mh_FirstTime[idint]->SetOption("BOX");
  ss.str(""); // clear
  }

  if(flag==101) {
  ss <<idint<<"_Anode_Chan_Eff";
  mh_ChanEff[idint]=new TH1F(ss.str().c_str(),"",675,0.0,675.0);
  mh_ChanEff[idint]->GetXaxis()->SetTitle("(AFEB-1)*16+ch");
  mh_ChanEff[idint]->GetYaxis()->SetTitle("Entries");
  ss.str(""); // clear
  }

  if(flag==200) { 
  ss <<idint<<"_Anode_AfebDac";
  mh_AfebDac[idint]=new TH2F(ss.str().c_str(),"",75,0.0,75.0, 50,0.0,50.0);
  mh_AfebDac[idint]->GetXaxis()->SetTitle("Threshold DAC");
  mh_AfebDac[idint]->GetYaxis()->SetTitle("AFEB Channel Occupancy");
  mh_AfebDac[idint]->SetOption("COL");
  ss.str(""); // clear
  }

  if(flag==300) { 
  ss <<idint<<"_Anode_AfebThrPar";
  mh_AfebThrPar[idint]=new TH2F(ss.str().c_str(),"",700,0.0,700.0, 50,0.0,50.0);
  mh_AfebThrPar[idint]->GetXaxis()->SetTitle("(AFEB-1)*16+ch");
  mh_AfebThrPar[idint]->GetYaxis()->SetTitle("AFEB Channel Threshold (DAC)");
  mh_AfebThrPar[idint]->SetOption("BOX");
  ss.str(""); // clear
  }

  if(flag==400) { 
  ss <<idint<<"_Anode_AfebNoisePar";
  mh_AfebNoisePar[idint]=new TH2F(ss.str().c_str(),"",700,0.0,700.0, 50,0.0,5.0);
  mh_AfebNoisePar[idint]->GetXaxis()->SetTitle("(AFEB-1)*16+ch");
  mh_AfebNoisePar[idint]->GetYaxis()->SetTitle("AFEB Channel Noise (DAC)");
  mh_AfebNoisePar[idint]->SetOption("BOX");
  ss.str(""); // clear
  }

  if(flag==500) { 
  ss <<idint<<"_Anode_AfebNDF";
  mh_AfebNDF[idint]=new TH2F(ss.str().c_str(),"",700,0.0,700.0, 25,-5.0,20.0);
  mh_AfebNDF[idint]->GetXaxis()->SetTitle("(AFEB-1)*16+ch");
  mh_AfebNDF[idint]->GetYaxis()->SetTitle("AFEB Channel Fit NDF");
  mh_AfebNDF[idint]->SetOption("BOX");
  ss.str(""); // clear
  }

  if(flag==600) { 
  ss <<idint<<"_Anode_AfebChi2perNDF";
  mh_AfebChi2perNDF[idint]=new TH2F(ss.str().c_str(),"",700,0.0,700.0, 50,0.0,10.0);
  mh_AfebChi2perNDF[idint]->GetXaxis()->SetTitle("(AFEB-1)*16+ch");
  mh_AfebChi2perNDF[idint]->GetYaxis()->SetTitle("AFEB Channel Fit Chi2/NDF");
  mh_AfebChi2perNDF[idint]->SetOption("BOX");
  ss.str(""); // clear
  }

}


/* Analyze the hits */
void CSCAFEBThrAnalysis::analyze(const CSCWireDigiCollection& wirecltn) {

std::ostringstream ss;
std::map<int,std::vector<int> >::iterator intIt;
std::map<int, std::vector<std::vector<int> > >::iterator wiredacIt;

std::map<int,TH1*>::iterator h1;
std::map<int,TH2*>::iterator h2;
std::vector<int> vec; 
int afeb,ch;
float x,y;

m_wire_ev.clear();

/// Find DAC from event number nmbev

nmbev++;
indDac=(nmbev-1)/EvDac;
float dac=BegDac+StepDac*indDac;
if(vecDac.size() <= indDac) vecDac.push_back(dac);

//std::cout<<"  Event "<<nmbev;
//std::cout<<"  "<<indDac<<" "<<vecDac[indDac]<<std::endl;

//Anode wires

  CSCWireDigiCollection::DigiRangeIterator wiredetUnitIt;
  if(wirecltn.begin() == wirecltn.end())  nmbev_no_wire++; 

  if(wirecltn.begin() !=  wirecltn.end()) {

  for (wiredetUnitIt=wirecltn.begin();
       wiredetUnitIt!=wirecltn.end();
       ++wiredetUnitIt){

    const CSCDetId& id = (*wiredetUnitIt).first;

    const int idchamber=id.endcap()*10000 +id.station()*1000+
                        id.ring()*100 +id.chamber();
    const int idlayer  =id.endcap()*100000+id.station()*10000+
                        id.ring()*1000+id.chamber()*10+id.layer();

    //    std::cout<<idchamber<<" "<<idlayer<<std::endl;

    const int maxwire=csctoafeb.getMaxWire(id.station(),id.ring());
    std::vector<int> wireplane(maxwire,0);

    const CSCWireDigiCollection::Range& range = (*wiredetUnitIt).second;
    for (CSCWireDigiCollection::const_iterator digiIt =
           range.first; digiIt!=range.second; ++digiIt){

      int iwire=(*digiIt).getWireGroup();
      if(iwire<=maxwire) {
        if(wireplane[iwire-1]==0) {
          wireplane[iwire-1]=(*digiIt).getBeamCrossingTag()+1;
          ch=csctoafeb.getAfebCh(id.layer(),(*digiIt).getWireGroup());
	  afeb=csctoafeb.getAfebPos(id.layer(),(*digiIt).getWireGroup());

          /// Plot time bin of the first hit vs AFEB channels

          h2=mh_FirstTime.find(idchamber); 
          if (h2==mh_FirstTime.end()) {
            bookForId(100,idchamber,"");
            h2=mh_FirstTime.find(idchamber); 
          }
          x=(afeb-1)*16+ch;
          y=wireplane[iwire-1];
          h2->second->Fill(x,y,1.0);

          ///  Plot "efficiency" in first 100 pulses vs  AFEB channels

	  if(nmbev <=100) {
            h1=mh_ChanEff.find(idchamber); 
            if (h1==mh_ChanEff.end()) {
              bookForId(101,idchamber,"");
              h1=mh_ChanEff.find(idchamber); 
            }
            x=(afeb-1)*16+ch;
            h1->second->Fill(x,1.0);
	  }
        } // end if wireplane[iwire-1]==0
      }   // end if iwire<=csctoafeb.getMaxWire(id.station(),id.ring()
    }     // end of for digis in layer

    /// Store time bin of the first hit into map

    if(m_wire_ev.find(idlayer) == m_wire_ev.end()) 
      m_wire_ev[idlayer]=wireplane;

  }       // end of cycle on detUnit


  /// Accumulate hits into map of wires vs DAC

  for(intIt=m_wire_ev.begin(); intIt!=m_wire_ev.end(); ++intIt) {
    const int idwirev=(*intIt).first;
    const std::vector<int> wiretemp=(*intIt).second;

    wiredacIt=m_wire_dac.find(idwirev);
    std::vector<int> zer(1,0);

    if(wiredacIt==m_wire_dac.end()) {
      for(unsigned int i=0;i<(*intIt).second.size();i++)
	m_wire_dac[idwirev].push_back(zer);   
      wiredacIt=m_wire_dac.find(idwirev);      
    }

    int unsigned ndacsize=wiredacIt->second[0].size();
    for(unsigned int j=0;j<(indDac+1-ndacsize);j++)
         for(unsigned int i=0;i<(*intIt).second.size();i++)
	   wiredacIt->second[i].push_back(0);
    
    for(unsigned int i=0;i<(*intIt).second.size();i++)
      if((*intIt).second[i]>0) wiredacIt->second[i][indDac]=
			       wiredacIt->second[i][indDac]+1;          
  } // end of adding hits to the map of wire/DAC
  } // end of if wire collection not empty  
}   // end of void CSCAFEBThrAnalysis


void CSCAFEBThrAnalysis::done() {

  float x,y;

  std::map<int, std::vector<std::vector<int> > >::iterator mwiredacIt;
  std::map<int, std::vector<std::vector<float> > >::iterator mresfordbIt;
  std::vector<int>::iterator vecIt;
  std::map<int,TH2*>::iterator h2;

  std::cout<<"Events analyzed  "<<nmbev<<std::endl;
  std::cout<<"Events no anodes "<<nmbev_no_wire<<std::endl<<std::endl;

  std::vector<float> inputx;
  std::vector<float> inputy;

  std::vector<float> mypar(2, 0.0);
  std::vector<float> ermypar(2, 0.0);
  float ercorr, chisq, edm;
  int ndf,niter;
 
  int ch, afeb, idchmb;

  CSCFitAFEBThr * fitAnodeThr;

  std::vector<float> fitres(4,0);


  for(mwiredacIt=m_wire_dac.begin();mwiredacIt!=m_wire_dac.end();++mwiredacIt){
    int idwiredac=(*mwiredacIt).first;

    int layer=idwiredac-(idwiredac/10)*10;   
    idchmb=idwiredac/10;

    for(int unsigned iwire=0; iwire<mwiredacIt->second.size();iwire++) {

       afeb=csctoafeb.getAfebPos(layer,iwire+1);
       ch=csctoafeb.getAfebCh(layer,iwire+1);
       int afebid=(idwiredac/10)*100+csctoafeb.getAfebPos(layer,iwire+1);

       h2=mh_AfebDac.find(afebid); 
         if(h2==mh_AfebDac.end()) {
           bookForId(200,afebid,"");
           h2=mh_AfebDac.find(afebid);
       }

       indDac=0;
       for(vecIt=mwiredacIt->second[iwire].begin();
          vecIt!=mwiredacIt->second[iwire].end(); ++vecIt) {

          x=vecDac[indDac]; y=*vecIt; 
          h2->second->Fill(x,y);

          inputx.push_back(x);
          inputy.push_back(y);

          indDac++;
       }    
       // end of DAC cycle to form vectors of input data (inputx,inputy)for fit
 

       //    std::cout<<afebid<<" "<<ch<<std::endl;
       //    for(unsigned int i=0;i<inputx.size();i++)
       //       std::cout<<" "<<inputy[i];
       //    std::cout<<std::endl;


  for(unsigned int i=0;i<2;i++) {mypar[i]=0.0; ermypar[i]=0.0;}
  ercorr=0.0; 
  chisq=0.0; 
  ndf=0; 
  niter=0; 
  edm=0.0;

  /// Fitting threshold curve
  fitAnodeThr=new CSCFitAFEBThr();
  fitAnodeThr->ThresholdNoise(inputx,inputy,npulses,mypar,ermypar,ercorr,chisq,ndf,niter,edm);
  delete fitAnodeThr;

  //  std::cout<<"Fit "<<mypar[0]<<" "<<mypar[1]<<" "<<ndf<<" "<<chisq
  //           <<std::endl<<std::endl;

  /// Histogram fit results for given CSC vs wire defined as x=(afeb-1)*16+ch 

    x=(afeb-1)*16+ch;

    /// Threshold 
    h2=mh_AfebThrPar.find(idchmb); 
    if(h2==mh_AfebThrPar.end()) {
      bookForId(300,idchmb,"");
      h2=mh_AfebThrPar.find(idchmb);
    }
    y=mypar[0];
    h2->second->Fill(x,y);

    /// Noise 
    h2=mh_AfebNoisePar.find(idchmb); 
    if(h2==mh_AfebNoisePar.end()) {
      bookForId(400,idchmb,"");
      h2=mh_AfebNoisePar.find(idchmb);
    }
    y=mypar[1];
    h2->second->Fill(x,y);

    /// NDF 
    h2=mh_AfebNDF.find(idchmb); 
    if(h2==mh_AfebNDF.end()) {
      bookForId(500,idchmb,"");
      h2=mh_AfebNDF.find(idchmb);
    }
    y=ndf;
    h2->second->Fill(x,y);

    /// Chi2/NDF 
    h2=mh_AfebChi2perNDF.find(idchmb); 
    if(h2==mh_AfebChi2perNDF.end()) {
      bookForId(600,idchmb,"");
      h2=mh_AfebChi2perNDF.find(idchmb);
    }
    y=0.0;
    if(ndf>0) y=chisq/(float)ndf;
    h2->second->Fill(x,y);

    /// Prepare vector of fit results 
    fitres[0]=mypar[0];
    fitres[1]=mypar[1];
    fitres[2]=ndf;
    fitres[3]=0.0;
    if(ndf>0) fitres[3]=chisq/(float)ndf;
    
    /// Store fit results to map of wire vectors of vectors of results 
    
    mresfordbIt=m_res_for_db.find(idwiredac);
    if(mresfordbIt==m_res_for_db.end())
      m_res_for_db[idwiredac].push_back(fitres);
    else m_res_for_db[idwiredac].push_back(fitres);

    inputx.clear();
    inputy.clear();

} // end for(int iwire=0)
}               // end of iteration thru m_wire_dac map
  
  std::cout<<"Size of map for DB "<<m_res_for_db.size()<<std::endl;
  
  std::cout<<"The following CSCs went to DB"<<std::endl<<std::endl;

  for(mresfordbIt=m_res_for_db.begin(); mresfordbIt!=m_res_for_db.end(); 
      ++mresfordbIt) {
      int idlayer=(*mresfordbIt).first;
      idchmb=idlayer/10;
      int layer=idlayer-idchmb*10;
	std::cout<<"CSC "<<idchmb<<"  Layer "<<layer<<"  "
                 <<(*mresfordbIt).second.size()<<std::endl;          
  }
  /*
  for(mresfordbIt=m_res_for_db.begin(); mresfordbIt!=m_res_for_db.end(); 
      ++mresfordbIt) {
      int idlayer=(*mresfordbIt).first;
      for (int i=0;i<(*mresfordbIt).second.size();i++) { 
	std::cout<<idlayer<<" "<<i+1<<"    ";
        for(int j=0;j<4;j++)
	  std::cout<< (*mresfordbIt).second[i][j]<<" ";
	std::cout<<std::endl;
      }
  }
  */
  if(hist_file!=0) { // if there was a histogram file...
    hist_file->Write(); // write out the histrograms
    delete hist_file; // close and delete the file
    hist_file=0; // set to zero to clean up
    std::cout << "Hist. file was closed\n";
  }

  std::cout<<" End of CSCAFEBThrAnalysis"<<std::endl;  
}
