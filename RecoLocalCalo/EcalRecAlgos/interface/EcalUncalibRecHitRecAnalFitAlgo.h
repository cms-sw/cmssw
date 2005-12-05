#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAnalFitAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecAnalFitAlgo_HH

/** \class EcalUncalibRecHitRecAnalFitAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using an analytical fit
  *
  *  $Id: EcalUncalibRecHitRecAnalFitAlgo.h,v 1.1 2005/11/28 16:38:26 rahatlou Exp $
  *  $Date: 2005/11/28 16:38:26 $
  *  $Revision: 1.1 $
  *  \author A. Palma, Sh. Rahatlou Roma1
  */

#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/SymMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include <vector>
#include <string>

#include "TROOT.h"
#include "TMinuit.h"
#include "TGraph.h"
#include "TF1.h"

template<class C> class EcalUncalibRecHitRecAnalFitAlgo : public EcalUncalibRecHitRecAbsAlgo<C>
{
 public:
  // destructor
  virtual ~EcalUncalibRecHitRecAnalFitAlgo<C>() { };

  /// Compute parameters
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const std::vector<double>& pedestals,
                                            const std::vector<HepMatrix>& weights,
                                            const std::vector<HepSymMatrix>& chi2Matrix) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);

    // Get time samples
    //HepMatrix frame(C::MAXSAMPLES, 1);
    double frame[C::MAXSAMPLES];
    int gainId0 = dataFrame.sample(0).gainId();
    int iGainSwitch = 0;
    double maxsample(-1);
    int imax(-1);

    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      frame[iSample] = double(dataFrame.sample(iSample).adc());
      if (dataFrame.sample(iSample).gainId() > gainId0) iGainSwitch = 1;
      if( frame[iSample]>maxsample ) {
          maxsample= frame[iSample];
          imax=iSample;
      }
    }


    // Compute parameters
    //std::cout << "EcalUncalibRecHitRecAnalFitAlgo::makeRecHit() not yey implemented. returning dummy rechit" << std::endl;

    //analytic fit section
    double  xarray[10]={0.,1.,2.,3.,4.,5.,6.,7.,8.,9.};
    TGraph *graph=new TGraph(10,xarray,frame);
    TF1 pulseShape = TF1("pulseShape","[0]*pow((x - [3])/[1],[2])*exp(-[2]*(x - [1] - [3])/[1])",imax-1.,imax+3.);
    TF1 pedestal = TF1("pedestal","[0]",0.,2.);

    //pulseShape parameters
    double FIT_A=(double)maxsample;  //Amplitude
    double FIT_Tp=(double)imax;  //T peak
    double FIT_ALFA=1.5;  //Alpha
    double FIT_To=3.;  //T off
    // Amplitude
    pulseShape.SetParameter(0,FIT_A);
    // T peak
    pulseShape.SetParameter(1,FIT_Tp);
    // Alpha
    pulseShape.SetParameter(2,FIT_ALFA);
    // T off
    pulseShape.SetParameter(3,FIT_To);

    /*SINGLE XTAL TEST BEGIN
      if((itdg->id()).ieta()==85 && (itdg->id()).iphi()==19){
      TCanvas *canvas=new TCanvas("canvas","canvas",200,10,700,500); 
      graph->SetMarkerStyle(21);
      graph->SetMarkerColor(4);
      graph->Fit("pulseShape","QR"); 
      graph->Draw("AP");
      canvas->Update();
      canvas->Print("canvas.root");
      }
      else graph->Fit("pulseShape","QR"); 
      SINGLE XTAL TEST END*/

    graph->Fit("pulseShape","QRM");
    //TF1 *pulseShape2=graph->GetFunction("pulseShape");
    if ( std::string(gMinuit->fCstatu.Data()) == std::string("CONVERGED ") ) {

      double amplitude_value=pulseShape.GetParameter(0);

      graph->Fit("pedestal","QR");
      //TF1 *pedestal2=graph->GetFunction("pedestal");
      double pedestal_value=pedestal.GetParameter(0);

      amplitude_ = amplitude_value - pedestal_value;
      pedestal_  = pedestal_value;
      jitter_    = pulseShape.GetParameter(3);

    }

    delete graph;
    return EcalUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_);
  }
};
#endif
