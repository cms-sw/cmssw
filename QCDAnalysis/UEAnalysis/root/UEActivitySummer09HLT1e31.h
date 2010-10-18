#ifndef UEActivity_h
#define UEActivity_h

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TFile.h>

#include <TH1F.h>
#include <TH2D.h>
#include <TLorentzVector.h>
#include <TProfile.h>

#include <TClonesArray.h>

///
///_________________________________________________________
///
class UEActivity {

public :

  UEActivity();
  ~UEActivity() 
    {
      for ( unsigned int iregion(0); iregion<3; ++iregion )
        {
          h_pTChg      [iregion]->Delete();
          h_dN_vs_dphi [iregion]->Delete();
          h_dpT_vs_dphi[iregion]->Delete();
	  h_NChg       [iregion]->Delete();
	  h_pTSChg     [iregion]->Delete();
	  if(iregion==1)
	    {
	   h_dNMax_vs_dphi->Delete();
	   h_dNMin_vs_dphi->Delete();
	   h_dpTMax_vs_dphi->Delete();
	   h_dpTMin_vs_dphi->Delete();
	    }
	}
    }

  ///
  /// getter functions for UE activity
  ///
  TLorentzVector& GetLeadingJet()                    { return *_leadingLet;                       }
  double GetNumParticles( unsigned int iregion )     { return h_dN_vs_dphi[iregion]->Integral();  }
  double GetParticlePtSum( unsigned int iregion )    { return h_dpT_vs_dphi[iregion]->Integral(); }
  //new
  double GetNumParticleTraMax()                      { return h_dNMax_vs_dphi->Integral();        }
  double GetNumParticleTraMin()                      { return h_dNMin_vs_dphi->Integral();        }
  double GetPtSumParticleTraMax()                    { return h_dpTMax_vs_dphi->Integral();        }
  double GetPtSumParticleTraMin()                    { return h_dpTMin_vs_dphi->Integral();        }
 

  TH1D*  GetTH1D_pTChg( unsigned int iregion )       { return h_pTChg[iregion];                   }
  //new
  TH1D*  GetTH1D_NChg( unsigned int iregion )        { return h_NChg[iregion];                   }
  TH1D*  GetTH1D_pTSChg( unsigned int iregion )      { return h_pTSChg[iregion];                   }

  TH1D*  GetTH1D_dN_vs_dphi( unsigned int iregion )  { return h_dN_vs_dphi[iregion];              }
  TH1D*  GetTH1D_dpT_vs_dphi( unsigned int iregion ) { return h_dpT_vs_dphi[iregion];             }

  ///
  /// setter functions for UE activity
  ///
  void SetLeadingJet ( const TLorentzVector& theLeadingJet  ) 
    { 
      _leadingLet->SetPtEtaPhiE( theLeadingJet.Pt(), theLeadingJet.Eta(), theLeadingJet.Phi(), theLeadingJet.E() );
    }
  void SetTH1D_pTChg      ( unsigned int iregion, const TH1D* TH1D_pTChg       ) { h_pTChg      [iregion]->Add(TH1D_pTChg);       }
  void SetTH1D_dN_vs_dphi ( unsigned int iregion, const TH1D* TH1D_dN_vs_dphi  ) { h_dN_vs_dphi [iregion]->Add(TH1D_dN_vs_dphi);  }
  void SetTH1D_dpT_vs_dphi( unsigned int iregion, const TH1D* TH1D_dpT_vs_dphi ) { h_dpT_vs_dphi[iregion]->Add(TH1D_dpT_vs_dphi); } 
  //new
  void SetTH1D_NChg(unsigned int iregion, const TH1D* TH1D_NChg)                 {h_NChg        [iregion]->Add(TH1D_NChg);        }
  void SetTH1D_pTSChg (unsigned int iregion, const TH1D* TH1D_pTSChg)            {h_pTSChg      [iregion]->Add(TH1D_pTSChg);      }
  void SetTH1D_dNMax_vs_dphi (const TH1D* TH1D_dN_vs_dphi)                       {h_dNMax_vs_dphi->Add(TH1D_dN_vs_dphi);          }
  void SetTH1D_dNMin_vs_dphi (const TH1D* TH1D_dN_vs_dphi)                       {h_dNMin_vs_dphi->Add(TH1D_dN_vs_dphi);          }
  void SetTH1D_dpTMax_vs_dphi (const TH1D* TH1D_dpT_vs_dphi)                     {h_dpTMax_vs_dphi->Add(TH1D_dpT_vs_dphi);        }
  void SetTH1D_dpTMin_vs_dphi (const TH1D* TH1D_dpT_vs_dphi)                     {h_dpTMin_vs_dphi->Add(TH1D_dpT_vs_dphi);        }

private :

  double _etaRegion;
  double _ptThreshold;

  TLorentzVector* _leadingLet;
  TH1D**  h_pTChg;
  TH1D**  h_dN_vs_dphi;
  TH1D**  h_dpT_vs_dphi;
  //new
  TH1D**  h_NChg;
  TH1D**  h_pTSChg;
 
  TH1D* h_dNMax_vs_dphi;
  TH1D* h_dNMin_vs_dphi;
  TH1D* h_dpTMax_vs_dphi;
  TH1D* h_dpTMin_vs_dphi;
};


///
///_________________________________________________________
///
class UEActivityFinder {

public :

  UEActivityFinder( double, double );
  ~UEActivityFinder() 
    {
      for ( unsigned int iregion(0); iregion<3; ++iregion )
	{
	  _h_pTChg      [iregion]->Delete();
	  _h_dN_vs_dphi [iregion]->Delete();
	  _h_dpT_vs_dphi[iregion]->Delete();
	  _h_NChg       [iregion]->Delete();
	  _h_pTSChg     [iregion]->Delete();
	  if(iregion==1)
	    {
	    _h_dN_vs_dphiN1->Delete();
	    _h_dN_vs_dphiN2->Delete();
	    _h_dpT_vs_dphiN1->Delete();
	    _h_dpT_vs_dphiN2->Delete();
	    }
	}
    }

  Bool_t find( TClonesArray&, TClonesArray&, UEActivity& );

private :

  double etaRegion;
  double ptThreshold;

  TH1D** _h_pTChg;
  TH1D** _h_dN_vs_dphi;
  TH1D** _h_dpT_vs_dphi;
  //new
  TH1D** _h_NChg;
  TH1D** _h_pTSChg;
  TH1D* _h_dN_vs_dphiN1;
  TH1D* _h_dN_vs_dphiN2;
  TH1D* _h_dpT_vs_dphiN1;
  TH1D* _h_dpT_vs_dphiN2;
};


///
///_________________________________________________________
///
class UEActivityHistograms {

public :

  UEActivityHistograms( const char*, std::string* );
  ~UEActivityHistograms() { file->Write(); file->Close(); }; 

  void fill( UEActivity& );
  void fill( UEActivity&, TClonesArray& );

 private :

  TFile* file;

  std::string HLTBitNames[12];

  unsigned int _nbinsDphi, _nbinsPtChg;
  double       _xminDphi, _xminPtChg;
  double       _xmaxDphi, _xmaxPtChg;
  double       _binwidthDphi, _binwidthPtChg;

  TH1D** h_pTJet;
  TH1D** h_etaJet;
  TH2D** h_pTChg; 
  TH2D** h_dN_vs_dphi; 
  TH2D** h_dpT_vs_dphi; 
  TH2D** h_dN_vs_dpTjet; 
  TH2D** h_dpT_vs_dpTjet; 
  TH2D** h_averagePt_vs_nChg;
  //new
  TH1D **h_NChg;
  TH1D **h_pTSChg;

  TH2D** h_dNMax_vs_dpTjet;
  TH2D** h_dNMin_vs_dpTjet;
  TH2D** h_dpTMax_vs_dpTjet;
  TH2D** h_dpTMin_vs_dpTjet;
};

#endif
