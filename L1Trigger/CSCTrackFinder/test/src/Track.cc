#include "L1Trigger/CSCTrackFinder/test/src/Track.h"
#include <iostream>
using namespace std;
namespace csctf_analysis
{
Track::Track()
{
      matched = false;
      matchedIndex = -1;
      R = 20;
      ghost = false;
      TFPt=-1;
      Quality=-1;
      ghostMatchedToIndex = new std::vector<unsigned int>;
      ghostR = new std::vector<double>;
      ghostQ = new std::vector<double>;
}
Track::Track(const Track& track)
{
      ghostMatchedToIndex = new std::vector<unsigned int>;
      ghostR = new std::vector<double>;
      ghostQ = new std::vector<double>;
      *ghostMatchedToIndex= *track.ghostMatchedToIndex;
      *ghostR = *track.ghostR;
      *ghostQ = *track.ghostQ;
      matched=track.matched;
      matchedIndex=track.matchedIndex;
      R=track.R;
      histlist=track.histlist;
      ghost=track.ghost;
      Quality=track.Quality;
      TFPt=track.TFPt;
}
Track::~Track() 
{
      delete ghostMatchedToIndex;
      delete ghostR;
      delete ghostQ;
}
void Track::matchedTo(int i, double newR)
{
      if(matched==false || newR<R)
      {
              matched=true;
              R=newR;
              matchedIndex=i;
      }
}
void Track::unMatch()
{
      matched = false;
      matchedIndex = -1;
      R = 20;
      TFPt=-1;
      Quality=-1;
}
void Track::loseBestGhostCand(std::string param)
{
      int loseindex=0;
      int size=ghostMatchedToIndex->size();
      if(param == 'Q')        
      {
              for(int i=0;i<size;i++)
              {
        	      if (ghostQ->at(i)< ghostQ->at(loseindex))
        	      {       
        		      loseindex=i;
        	      }
              }
      }
      else if(param == 'R')   
      {
              for(int i=0;i<size;i++)
              {
        	      if(ghostR->at(i)>ghostR->at(loseindex))
        	      {
        		      loseindex=i;
        	      }
              }
      }
      else
      {
              std::cout <<"\nChoose Q or R for ghost Candidate parameter!!!\n";
      }
      ghostMatchedToIndex->erase(ghostMatchedToIndex->begin()+loseindex);
      ghostQ->erase(ghostQ->begin()+loseindex);
      ghostR->erase(ghostR->begin()+loseindex);
}
void Track::fillSimvTFHist(const Track& simtrack, const Track& tftrack) const
{
      double simfabsEta=fabs(simtrack.getEta());
      double simpt=simtrack.getPt();
      double tfpt=tftrack.getPt();        
      if(simfabsEta>=0.9) 
      {
              histlist->SimPt_vs_TFPt_FWD->Fill(simpt,tfpt);
      }
      if(simfabsEta<=0.9) 
      {
              histlist->SimPt_vs_TFPt_DT->Fill(simpt,tfpt);
      }
}
void Track::fillHist() 
{
      int qual=getQuality();
      double pt=getPt();
      double mode=getMode();
      double fabsEta=fabs(getEta());

      
      //for phi mod 10 deg histograms
      double phi_deg=getPhi()*180/3.1415927;
      double result=phi_deg-5;    
      if(result<0) {result+=360;}
      while(result>=10){result-=10.0;}
      double phi_bar=result;
      

      histlist->Eta->Fill(fabsEta);
      histlist->signedEta->Fill(getEta());

      histlist->Phi->Fill(getPhi());


      if(getEta()>0.9 && getEta()<2.4)
      {
              histlist->Phi_mod_10_endcap1->Fill(phi_bar);
      }       
      if(getEta()<-0.9 && getEta()>-2.4)
      {
              histlist->Phi_mod_10_endcap2->Fill(phi_bar);
      } 	    
      histlist->Pt->Fill(pt);    
      histlist->Pz->Fill(getPz());    
      histlist->P->Fill(getP());      
      histlist->Radius->Fill(getRadius());    
      histlist->Quality->Fill(qual);
      histlist->modeOcc->Fill(mode);
      histlist->FR->Fill(getFR());
      histlist->ptDenOverall->Fill(pt);

      //DT Region
      if( fabsEta >= 0 && fabsEta <= 0.9 )// getEta() changed to fabsEta
      {
              histlist->ptDenDTOnly->Fill(pt);
              //histlist->fidPtDen->Fill(pt);
              histlist->modeOccDT->Fill(mode);
      }
      //CSC Only
      if( fabsEta >= 1.2 && fabsEta <= 2.4)
      {
              histlist->ptDenCSCOnly->Fill(pt);
              histlist->modeOccCSCOnly->Fill(mode);
      }
      //CSC Restricted
      if( fabsEta >= 1.2 && fabsEta <= 2.1)
      {
              histlist->ptDenCSCRestricted->Fill(pt);
      }       
      //Overlap
      if( fabsEta <= 1.2 && fabsEta >= 0.9)
      {
              histlist->ptDenOverlap->Fill(pt);       
              histlist->modeOccOverlap->Fill(mode);
      }
      if(fabsEta >= 2.1)
      {
              histlist->ptDenHighEta->Fill(pt);       
              histlist->modeOccHighEta->Fill(mode);
      }
}
void Track::fillMatchHist() 
{
      int i=0;
      int qual=getQuality();
      double tfpt=getTFPt();
      double simpt=getPt();
      double mode=getMode();
      double fabsEta = fabs(getEta());
      double phi_deg = getPhi()*180/3.1415926;        
      double result=phi_deg-5;
      if(result<0)
      {
              result+=360;
      }
      while(result>=10)
      {
              result-=10.0;
              i++;
      }
      double phi_bar=result;
      if(fabsEta>=0.9) 
      {
              histlist->matchedRefPt_FWD->Fill(simpt);
      }
      if(fabsEta<=0.9) 
      {
              histlist->matchedRefPt_DT->Fill(simpt);
      }
      histlist->matchEta->Fill(fabsEta);      
      histlist->signedMatchEta->Fill(getEta());       
      histlist->matchPhi->Fill(getPhi());
      histlist->matchRadius->Fill(getRadius());
      histlist->matchMode->Fill(mode);       
      if(qual>0)
      {
              histlist->EtaQ1->Fill(fabsEta); 
              histlist->signedEtaQ1->Fill(getEta());  
              histlist->PhiQ1->Fill(getPhi());        
              histlist->PtQ1->Fill(simpt);  
      }
      if(qual>1)
      {
              histlist->EtaQ2->Fill(fabsEta); 
              histlist->signedEtaQ2->Fill(getEta());  
              histlist->PhiQ2->Fill(getPhi());        
              histlist->PtQ2->Fill(simpt);  
              if(getEta()>0.9&&getEta()<2.4)
              {
        	      histlist->matchPhi_mod_10_Q2_endcap1->Fill(phi_bar);
              }       
              if(getEta()<-0.9&&getEta()>-2.4)
              {
        	      histlist->matchPhi_mod_10_Q2_endcap2->Fill(phi_bar);
              }       
      }
      if(qual>2)
      {
              histlist->EtaQ3->Fill(fabsEta); 
              histlist->signedEtaQ3->Fill(getEta());  
              histlist->PhiQ3->Fill(getPhi());        
              histlist->PtQ3->Fill(simpt);  
              if(getEta()>0.9 && getEta()<2.4)
              {
        	      histlist->matchPhi_mod_10_Q3_endcap1->Fill(phi_bar);
              }       
              if(getEta()<-0.9 && getEta()>-2.4)
              {
        	      histlist->matchPhi_mod_10_Q3_endcap2->Fill(phi_bar);
              }
      }
      ///////////////////////////
      //  Section filling pt
      ///////////////////////////
      //Overall
      
      if(qual>1)
      {
      	      	histlist->matchPtOverall->Fill(simpt);  
              	if (tfpt>10)
              	{
        		histlist->matchTFPt10Overall->Fill(simpt);
              	}
	      	if (tfpt>12)
              	{
        	     	histlist->matchTFPt12Overall->Fill(simpt);
              	}
	      	if (tfpt>16)
              	{
        	      	histlist->matchTFPt16Overall->Fill(simpt);
              	}
              	if (tfpt>20)
    	        {
        	      	histlist->matchTFPt20Overall->Fill(simpt);
              	}
              	if (tfpt>40)
              	{
        	      	histlist->matchTFPt40Overall->Fill(simpt);
              	}
              	if (tfpt>60)
              	{
        	      	histlist->matchTFPt60Overall->Fill(simpt);
              	}
      }
      //DT Region
      if(fabsEta>=0&&fabsEta<=0.9&&qual>1)
      { 
              //histlist->matchPt->Fill(simpt);
              histlist->matchPtDTOnly->Fill(simpt);   
              if (tfpt>10)
              {
        	      histlist->matchTFPt10DTOnly->Fill(simpt);
              }
	      if (tfpt>12)
              {
        	      histlist->matchTFPt12DTOnly->Fill(simpt);
              }
	      if (tfpt>16)
              {
        	      histlist->matchTFPt16DTOnly->Fill(simpt);
              }
              if (tfpt>20)
              {
        	      histlist->matchTFPt20DTOnly->Fill(simpt);
              }
              if (tfpt>40)
              {
        	      histlist->matchTFPt40DTOnly->Fill(simpt);
              }
              if (tfpt>60)
              {
        	      histlist->matchTFPt60DTOnly->Fill(simpt);
              }
      }
      //CSC Only
      if(fabsEta>=1.2&&fabsEta<=2.4&&qual>1)
      {
              histlist->matchPtCSCOnly->Fill(simpt);  
              if (tfpt>10)
              {
        	      histlist->matchTFPt10CSCOnly->Fill(simpt);
              }
	      if (tfpt>12)
              {
        	      histlist->matchTFPt12CSCOnly->Fill(simpt);
              }
	      if (tfpt>16)
              {
        	      histlist->matchTFPt16CSCOnly->Fill(simpt);
              }
              if (tfpt>20)
              {
        	      histlist->matchTFPt20CSCOnly->Fill(simpt);
              }
              if (tfpt>40)
              {
        	      histlist->matchTFPt40CSCOnly->Fill(simpt);
              }
              if (tfpt>60)
              {
        	      histlist->matchTFPt60CSCOnly->Fill(simpt);
              }
      }
      //CSC Restricted
      if(fabsEta>=1.2&&fabsEta<=2.1&&qual>1)
      {
              histlist->matchPtCSCRestricted->Fill(simpt);    
              if (tfpt>10)
              {
        	      histlist->matchTFPt10CSCRestricted->Fill(simpt);
              }
	      if (tfpt>12)
              {
        	      histlist->matchTFPt12CSCRestricted->Fill(simpt);
              }
	      if (tfpt>16)
              {
        	      histlist->matchTFPt16CSCRestricted->Fill(simpt);
              }
              if (tfpt>20)
              {
        	      histlist->matchTFPt20CSCRestricted->Fill(simpt);
              }
              if (tfpt>40)
              {
        	      histlist->matchTFPt40CSCRestricted->Fill(simpt);
              }
              if (tfpt>60)
              {
        	      histlist->matchTFPt60CSCRestricted->Fill(simpt);
              }
      }
      //Overlap
      if(fabsEta<=1.2&&fabsEta>=0.9&&qual>1)
      {
              histlist->matchPtOverlap->Fill(simpt);  
              if (tfpt>10)
              {
        	      histlist->matchTFPt10Overlap->Fill(simpt);
              }
	      if (tfpt>12)
              {
        	      histlist->matchTFPt12Overlap->Fill(simpt);
              }
	      if (tfpt>16)
              {
        	      histlist->matchTFPt16Overlap->Fill(simpt);
              }
              if (tfpt>20)
              {
        	      histlist->matchTFPt20Overlap->Fill(simpt);
              }
              if (tfpt>40)
              {
        	      histlist->matchTFPt40Overlap->Fill(simpt);
              }
              if (tfpt>60)
              {
        	      histlist->matchTFPt60Overlap->Fill(simpt);
              }
      }
      //High Eta
      if(fabsEta>=2.1&&qual>1)
      {
              histlist->matchPtHighEta->Fill(simpt);  
              if (tfpt>10)
              {
        	      histlist->matchTFPt10HighEta->Fill(simpt);
              }
	      if (tfpt>12)
              {
        	      histlist->matchTFPt12HighEta->Fill(simpt);
              }
	      if (tfpt>16)
              {
        	      histlist->matchTFPt16HighEta->Fill(simpt);
              }
              if (tfpt>20)
              {
        	      histlist->matchTFPt20HighEta->Fill(simpt);
              }
              if (tfpt>40)
              {
        	      histlist->matchTFPt40HighEta->Fill(simpt);
              }
              if (tfpt>60)
              {
        	      histlist->matchTFPt60HighEta->Fill(simpt);
              }
      }
      
}
  void Track::fillGhostHist()
{
      int qual=getQuality();
      double Eta=getEta();
      double fabsEta=fabs(Eta);
      double simpt=getPt(); 
      double Phi=getPhi();
      histlist->ghostEta->Fill(fabsEta);      
      histlist->ghostSignedEta->Fill(Eta);       
      histlist->ghostPhi->Fill(Phi);     
      histlist->ghostPt->Fill(simpt);       
      histlist->ghostRadius->Fill(getRadius());       
      //histlist->ghostQuality->Fill(qual);     

      if (qual>0)
      {
              histlist->ghostEtaQ1->Fill(fabsEta);    
              histlist->ghostSignedEtaQ1->Fill(Eta);     
              histlist->ghostPhiQ1->Fill(Phi);   
              histlist->ghostPtQ1->Fill(simpt);     
      }
      if (qual>1)
      {
              histlist->ghostEtaQ2->Fill(fabsEta);    
              histlist->ghostSignedEtaQ2->Fill(Eta);     
              histlist->ghostPhiQ2->Fill(Phi);   
              histlist->ghostPtQ2->Fill(simpt);     
      }
      if (qual>2)
      {
              histlist->ghostEtaQ3->Fill(fabsEta);    
              histlist->ghostSignedEtaQ3->Fill(Eta);     
              histlist->ghostPhiQ3->Fill(Phi);   
              histlist->ghostPtQ3->Fill(simpt);     
      }
  }
  void Track::fillRateHist()
  {
  	double pt=getPt()+0.001; //addition of 0.001 is to ensure the Pt is not on the border of a bin- which causes problems with the iterative stepping through bins
	double stepPt=histlist->getPtStep();
	
	for(double threshold=pt;threshold>=-1;threshold-=stepPt)
	{
		histlist->rateHist->Fill(threshold);
	}
  
  
  }
}
