// -*- C++ -*-
// AUTHOR:  Anil Singh (anil@cern.ch), Lovedeep Saini (lovedeep@cern.ch)

#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/VetoedFinalState.hh"
#include "Rivet/Projections/InvMassFinalState.hh"
#include "Rivet/Tools/ParticleIdUtils.hh"


namespace Rivet {
  
  
  class CMS_EWK_10_012 : public Analysis {
  public:
    
    CMS_EWK_10_012()
      : Analysis("CMS_EWK_10_012")
    {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }
    
    
    /// Book histograms and initialise projections before the run
    void init() {
      
      const FinalState fs(-MAXRAPIDITY,MAXRAPIDITY);
      addProjection(fs, "FS");
      
      vector<pair<PdgId,PdgId> > vidsZ;
      vidsZ.push_back(make_pair(ELECTRON, POSITRON));
      vidsZ.push_back(make_pair(MUON, ANTIMUON));

      FinalState fsZ(-MAXRAPIDITY,MAXRAPIDITY);
      InvMassFinalState invfsZ(fsZ, vidsZ, 60*GeV, 120*GeV);
      addProjection(invfsZ, "INVFSZ");
      
      vector<pair<PdgId,PdgId> > vidsW;
      vidsW.push_back(make_pair(ELECTRON, NU_EBAR));
      vidsW.push_back(make_pair(POSITRON, NU_E));
      vidsW.push_back(make_pair(MUON, NU_MUBAR));
      vidsW.push_back(make_pair(ANTIMUON, NU_MU));
      
      FinalState fsW(-MAXRAPIDITY,MAXRAPIDITY);
      InvMassFinalState invfsW(fsW, vidsW, 20*GeV, 99999*GeV);
      addProjection(invfsW, "INVFSW");
      
      VetoedFinalState vfs(fs);
      vfs.addVetoOnThisFinalState(invfsZ);
      vfs.addVetoOnThisFinalState(invfsW);
      addProjection(vfs, "VFS");
      addProjection(FastJets(vfs, FastJets::ANTIKT, 0.5), "Jets");
      
      _histNoverN0Welec = bookDataPointSet(1,1,1);   
      _histNoverNm1Welec = bookDataPointSet(2,1,1);   
      _histNoverN0Wmu = bookDataPointSet(3,1,1);
      _histNoverNm1Wmu = bookDataPointSet(4,1,1);   
      _histNoverN0Zelec = bookDataPointSet(5,1,1);   
      _histNoverNm1Zelec = bookDataPointSet(6,1,1);   
      _histNoverN0Zmu = bookDataPointSet(7,1,1);   
      _histNoverNm1Zmu = bookDataPointSet(8,1,1);   
      _histJetMultWelec  = bookHistogram1D("njetWenu", 5, -0.5, 4.5);
      _histJetMultWmu    = bookHistogram1D("njetWmunu", 5, -0.5, 4.5);
      _histJetMultZelec  = bookHistogram1D("njetZee", 5, -0.5, 4.5);
      _histJetMultZmu    = bookHistogram1D("njetZmumu", 5, -0.5, 4.5);

      _histJetMultWmuPlus = bookHistogram1D("njetWmuPlus", 5, -0.5, 4.5);
      _histJetMultWmuMinus = bookHistogram1D("njetWmuMinus", 5, -0.5, 4.5);
      _histJetMultWelPlus = bookHistogram1D("njetWePlus", 5, -0.5, 4.5);
      _histJetMultWelMinus = bookHistogram1D("njetWeMinus", 5, -0.5, 4.5);
        _histJetMultRatioWmuPlusMinus = bookDataPointSet(10, 1, 1);
	_histJetMultRatioWelPlusMinus = bookDataPointSet(9, 1, 1);

    } 

    
    bool ApplyElectronCutsForZee(double pt1, double pt2, double eta1, double eta2){
      bool isFid1 = ((fabs(eta1)<1.4442)||((fabs(eta1)>1.566)&&(fabs(eta1)<2.5)));
      bool isFid2 = ((fabs(eta2)<1.4442)||((fabs(eta2)>1.566)&&(fabs(eta2)<2.5)));
      if( isFid1 && isFid2 && pt1>20 && pt2 >10) return true;
      else return false;
    }

    
    bool ApplyMuonCutsForZmm(double pt1, double pt2, double eta1, double eta2){
      bool isFid1 = ((fabs(eta1)<2.1));
      bool isFid2 = ((fabs(eta2)<2.4));
      if( isFid1 && isFid2 && pt1>20 && pt2 >10) return true;
      else return false;
    }


    bool ApplyElectronCutsForWen(double pt1, double eta1){
      bool isFid1 = ((fabs(eta1)<1.4442)||((fabs(eta1)>1.566)&&(fabs(eta1)<2.5)));
      if( isFid1 && pt1>20 ) return true;
      return 0;
    }
 
   
    bool ApplyMuonCutsForWmn(double pt1, double eta1){
      bool isFid1 = ((fabs(eta1)<2.1));
      if( isFid1 && pt1>20) return true;
      return 0;
    }
    

    void Fill(AIDA::IHistogram1D*& _histJetMult, const double& weight, std::vector<FourMomentum>& finaljet_list){
      _histJetMult->fill(0, weight);
      for (size_t i=0 ; i<finaljet_list.size() ; ++i) {
        if (i==6) break;
        _histJetMult->fill(i+1, weight);  // inclusive
      }
    }  
    
    void FillNoverNm1(AIDA::IHistogram1D*& _histJetMult,AIDA::IDataPointSet* _histNoverNm1){
      std::vector<double> y, yerr;
      for (int i=0; i<_histJetMult->axis().bins()-1; i++) {
        double val = 0.;
        double err = 0.;
        if (!fuzzyEquals(_histJetMult->binHeight(i), 0)) {
          val = _histJetMult->binHeight(i+1) / _histJetMult->binHeight(i);
          err = val * sqrt(  pow(_histJetMult->binError(i+1)/_histJetMult->binHeight(i+1), 2)
                           + pow(_histJetMult->binError(i)  /_histJetMult->binHeight(i)  , 2) );
        }
        y.push_back(val);
        yerr.push_back(err);
      }
      _histNoverNm1->setCoordinate(1, y, yerr);
    }    
    void FillNoverN0(AIDA::IHistogram1D*& _histJetMult,AIDA::IDataPointSet* _histNoverN0){
      std::vector<double> y, yerr;
      for (int i=0; i<_histJetMult->axis().bins()-1; i++) {
        double val = 0.;
        double err = 0.;
        if (!fuzzyEquals(_histJetMult->binHeight(i), 0)) {
          val = _histJetMult->binHeight(i+1) / _histJetMult->binHeight(0);
          err = val * sqrt(  pow(_histJetMult->binError(i+1)/_histJetMult->binHeight(i+1), 2)
                           + pow(_histJetMult->binError(0)  /_histJetMult->binHeight(0)  , 2) );
        }
        y.push_back(val);
        yerr.push_back(err);
      }
      _histNoverN0->setCoordinate(1, y, yerr);
    }    

    
   void FillChargeAssymHistogramSet(  AIDA::IHistogram1D*& _histJetMult1,AIDA::IHistogram1D*& _histJetMult2, AIDA::IDataPointSet* _histJetMultRatio12 ){
      std::vector<double> yval, yerr;
      for (int i = 0; i < 4; ++i) {
        std::vector<double> xval; xval.push_back(i);
        std::vector<double> xerr; xerr.push_back(.5);
        double ratio = 0;
        double err = 0.;
        double num = _histJetMult1->binHeight(i)-_histJetMult2->binHeight(i);
	double den = _histJetMult1->binHeight(i)+_histJetMult2->binHeight(i);
	double errNum = 0;
	errNum = std::pow(_histJetMult1->binError(i),2)+std::pow(_histJetMult2->binError(i),2);
	double errDen = 0;
	errDen = std::pow(_histJetMult1->binError(i),2)+std::pow(_histJetMult2->binError(i),2); 

        if (den)ratio = num/den;

        if(num)
	  errNum = errNum/(num*num); 
        if(den) 
	  errDen = errDen/(den*den);

        err = std::sqrt(errDen+errNum);
	if(!(err==err))err=0;
        yval.push_back(ratio);
        yerr.push_back(ratio*err);
        }
        _histJetMultRatio12->setCoordinate(1,yval,yerr);
      }
    



    void analyze(const Event& event) {
      //some flag definitions.
      bool isZmm =false;
      bool isZee =false;
      bool isWmn =false;
      bool isWen =false;
      bool isWmnMinus =false;
      bool isWmnPlus  =false;
      bool isWenMinus =false;
      bool isWenPlus  =false;

      const double weight = event.weight();
      
      const InvMassFinalState& invMassFinalStateZ = applyProjection<InvMassFinalState>(event, "INVFSZ");
      const InvMassFinalState& invMassFinalStateW = applyProjection<InvMassFinalState>(event, "INVFSW");
      
      bool isW(false); bool isZ(false);
      
      isW  = (invMassFinalStateZ.empty() && !(invMassFinalStateW.empty()));
      isZ  = (!(invMassFinalStateZ.empty()) && invMassFinalStateW.empty());

      const ParticleVector&  ZDecayProducts =  invMassFinalStateZ.particles();
      const ParticleVector&  WDecayProducts =  invMassFinalStateW.particles();

      if (ZDecayProducts.size() < 2 && WDecayProducts.size() <2) vetoEvent;
      
      double pt1=-9999.,  pt2=-9999.;
      double phi1=-9999., phi2=-9999.;
      double eta1=-9999., eta2=-9999.;
      
      double mt = 999999;
      if(isZ){
	pt1  = ZDecayProducts[0].momentum().pT();
	pt2  = ZDecayProducts[1].momentum().pT();
	eta1 = ZDecayProducts[0].momentum().eta();
	eta2 = ZDecayProducts[1].momentum().eta();
	phi1 = ZDecayProducts[0].momentum().phi();
	phi2 = ZDecayProducts[1].momentum().phi();
      }
      
      if(isW){
	if(
	   (fabs(WDecayProducts[1].pdgId()) == NU_MU) || (fabs(WDecayProducts[1].pdgId()) == NU_E)){
	  pt1  = WDecayProducts[0].momentum().pT();
	  pt2  = WDecayProducts[1].momentum().Et();
          eta1 = WDecayProducts[0].momentum().eta();
          eta2 = WDecayProducts[1].momentum().eta();
	  phi1 = WDecayProducts[0].momentum().phi();
	  phi2 = WDecayProducts[1].momentum().phi();
          mt=sqrt(2.0*pt1*pt2*(1.0-cos(phi1-phi2)));
	}
	else {
	  pt1  = WDecayProducts[1].momentum().pT();
	  pt2  = WDecayProducts[0].momentum().Et();
          eta1 = WDecayProducts[1].momentum().eta();
          eta2 = WDecayProducts[0].momentum().eta();
	  phi1 = WDecayProducts[1].momentum().phi();
	  phi2 = WDecayProducts[0].momentum().phi();
          mt=sqrt(2.0*pt1*pt2*(1.0-cos(phi1-phi2)));
	}
      }

      if(isW && mt<20)vetoEvent;
            
      isZmm = isZ && ((fabs(ZDecayProducts[0].pdgId()) == 13) && (fabs(ZDecayProducts[1].pdgId()) == 13));
      isZee = isZ && ((fabs(ZDecayProducts[0].pdgId()) == 11) && (fabs(ZDecayProducts[1].pdgId()) == 11));
      isWmn  = isW && ((fabs(WDecayProducts[0].pdgId()) == 14) || (fabs(WDecayProducts[1].pdgId()) == 14));
      isWen  = isW && ((fabs(WDecayProducts[0].pdgId()) == 12) || (fabs(WDecayProducts[1].pdgId()) == 12));
      
      if(isWmn){
        if((WDecayProducts[0].pdgId()==-13)|| (WDecayProducts[1].pdgId()==-13)){
	isWmnMinus = false;
	isWmnPlus = true;
	  }	
       else{
	isWmnMinus = true;
	isWmnPlus  = false;
      } 
     }

      if(isWen){
       if((WDecayProducts[0].pdgId()==11)|| (WDecayProducts[1].pdgId()==11)){
	isWenMinus = true;
	isWenPlus = false;
      }
      else{
	isWenMinus = false;
	isWenPlus  = true;
       }
      }

      if(!((isZmm||isZee)||(isWmn||isWen)))vetoEvent;
              
      bool passBosonConditions = false;
      if(isZmm)passBosonConditions = ApplyMuonCutsForZmm(pt1,pt2,eta1,eta2);
      if(isZee)passBosonConditions = ApplyElectronCutsForZee(pt1,pt2,eta1,eta2);
      if(isWen)passBosonConditions = ApplyElectronCutsForWen(pt1,eta1);  
      if(isWmn)passBosonConditions = ApplyMuonCutsForWmn(pt1,eta1);  
      
      if(!passBosonConditions)vetoEvent;
          
      //Obtain the jets.
      vector<FourMomentum> finaljet_list;
      foreach (const Jet& j, applyProjection<FastJets>(event, "Jets").jetsByPt(30.0*GeV)) {
	const double jeta = j.momentum().eta();
	const double jphi = j.momentum().phi();
	const double jpt = j.momentum().pT();
	if (fabs(jeta) < 2.4) 
	  if(jpt>30){
	      if(isZee){
		  if (deltaR(pt1, phi1, jeta, jphi) > 0.3 && deltaR(pt2, phi2, jeta, jphi) > 0.3)
		    finaljet_list.push_back(j.momentum());
		  continue;
		}
	      else if(isWen){
		  if (deltaR(pt1, phi1, jeta, jphi) > 0.3)
		    finaljet_list.push_back(j.momentum());
		  continue;
	      }
	      
	      else  finaljet_list.push_back(j.momentum());
	  }
      }

      //Multiplicity plots.	
      if(isWen)Fill(_histJetMultWelec, weight, finaljet_list);
      if(isWmn)Fill(_histJetMultWmu, weight, finaljet_list);
      if(isWmnPlus)Fill(_histJetMultWmuPlus, weight, finaljet_list);
      if(isWmnMinus)Fill(_histJetMultWmuMinus, weight, finaljet_list);
      if(isWenPlus)Fill(_histJetMultWelPlus, weight, finaljet_list);
      if(isWenMinus)Fill(_histJetMultWelMinus, weight, finaljet_list);
      if(isZee)Fill(_histJetMultZelec, weight, finaljet_list);
      if(isZmm)Fill(_histJetMultZmu, weight, finaljet_list);
    }
    
    
    /// Normalise histograms etc., after the run
    void finalize() {
      FillNoverNm1(_histJetMultWelec,_histNoverNm1Welec);
      FillNoverN0(_histJetMultWelec,_histNoverN0Welec);
      FillNoverNm1(_histJetMultWmu,_histNoverNm1Wmu);
      FillNoverN0(_histJetMultWmu,_histNoverN0Wmu);
      FillNoverNm1(_histJetMultZelec,_histNoverNm1Zelec);
      FillNoverN0(_histJetMultZelec,_histNoverN0Zelec);
      FillNoverNm1(_histJetMultZmu,_histNoverNm1Zmu);
      FillNoverN0(_histJetMultZmu,_histNoverN0Zmu);
      FillChargeAssymHistogramSet(_histJetMultWmuPlus,_histJetMultWmuMinus, _histJetMultRatioWmuPlusMinus);
      FillChargeAssymHistogramSet(_histJetMultWelPlus,_histJetMultWelMinus, _histJetMultRatioWelPlusMinus);
    }

  private:

    AIDA::IHistogram1D*  _histJetMultWelec;
    AIDA::IDataPointSet* _histNoverNm1Welec;          // n/(n-1)
    AIDA::IDataPointSet* _histNoverN0Welec;          // n/n(0)
    
    AIDA::IHistogram1D*  _histJetMultWmu;
    AIDA::IDataPointSet* _histNoverNm1Wmu;          // n/(n-1)
    AIDA::IDataPointSet* _histNoverN0Wmu;          // n/n(0)

    AIDA::IHistogram1D*  _histJetMultWelMinus;
    AIDA::IHistogram1D*  _histJetMultWelPlus;
    AIDA::IDataPointSet* _histJetMultRatioWelPlusMinus;
    
    AIDA::IHistogram1D*  _histJetMultWmuMinus;
    AIDA::IHistogram1D*  _histJetMultWmuPlus;
    AIDA::IDataPointSet* _histJetMultRatioWmuPlusMinus;
   
    AIDA::IHistogram1D*  _histJetMultZelec;
    AIDA::IDataPointSet* _histNoverNm1Zelec;          // n/(n-1)
    AIDA::IDataPointSet* _histNoverN0Zelec;          // n/n(0)

    AIDA::IHistogram1D*  _histJetMultZmu;
    AIDA::IDataPointSet* _histNoverNm1Zmu;          // n/(n-1)
    AIDA::IDataPointSet* _histNoverN0Zmu;          // n/n(0)
  };
  
  AnalysisBuilder<CMS_EWK_10_012> plugin_CMS_EWK_10_012;
  
}

