#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"

namespace FitterFuncs{

   int cntNANinfit;
   double psFit_x[10], psFit_y[10], psFit_erry[10]; 
  // since we know we are counting in nanoseconds
  // we don't need to do an expensive finding of the bin
  // simply take floor(x) and determine if bin center is above or below
  // bin center is just bin + 0.5, inputs bins are 1ns wide
   float fast_interpolate(double x, const std::array<float,256>& h1) {
      if( x != x ){
         cntNANinfit ++;
         return h1[255];
      }

      const int bin = (int)x;

      if( x < 0.5 ) return h1[0];
      else if ( x > 255.5 ) return h1[255];

      const int bin_0 = ( x < bin+0.5 ? bin-1 : bin );
      const int bin_1 = ( x < bin+0.5 ? bin : bin+1 );

      const float slope = (h1[bin_1] - h1[bin_0])/(bin_1-bin_0);
      return h1[bin_0] + (x-bin_0-0.5f)*slope;
   }

   std::array<float,10> funcHPDShape(const std::vector<double>& pars,
                                     const std::array<float,256>& h1_single) {
    // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
      constexpr int ns_per_bx = 25;
      constexpr int num_ns = 250;
      constexpr int num_bx = num_ns/ns_per_bx;

    // zeroing output binned pulse shape
      std::array<float,num_bx> ntmpbin{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };

      for(int i=0;i < num_ns; ++i) {
         const float offset = i - 98.5f - pars[0]; // where does 98.5 come from?
         const float shifted_pulse1 = (offset < 0.0f ? 0.0f : fast_interpolate(offset,h1_single));
         ntmpbin[i/ns_per_bx] += shifted_pulse1;
      }
    // now we use ntmpbin to record the final pulse shape
      for(int i=0; i < num_bx; ++i) {
         ntmpbin[i] = pars[1]*ntmpbin[i] + pars[2];
      }
      return ntmpbin;
   }

   std::array<float,10> func_DoublePulse_HPDShape(const std::vector<double>& pars,
                                                  const std::array<float,256>& h1_double) {
    // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
      constexpr int ns_per_bx = 25;
      constexpr int num_ns = 250;
      constexpr int num_bx = num_ns/ns_per_bx;

    // zeroing output binned pulse shape
      std::array<float,num_bx> ntmpbin{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
      std::array<float,num_bx> ntmpbin2{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };

      for(int i=0;i < num_ns;++i) {
         const float offset1 = i - 98.5 - pars[0]; // where does 98.5 come from?
         const float offset2 = i - 98.5 - pars[1];

         ntmpbin[i/ns_per_bx] += (offset1 < 0.0f ? 0.0f : fast_interpolate(offset1,h1_double));
         ntmpbin2[i/ns_per_bx] += (offset2 < 0.0f ? 0.0f : fast_interpolate(offset2,h1_double));
      }
    // now we use ntmpbin to record the final pulse shape
      for(int i=0; i < num_bx; ++i) {
         ntmpbin[i] = pars[2]*ntmpbin[i]+pars[3]*ntmpbin2[i]+pars[4];
      }
      return ntmpbin;
   }

   SinglePulseShapeFunctor::SinglePulseShapeFunctor(const HcalPulseShapes::Shape& pulse) {
      for(int i=0;i<256;i++) {
         pulse_hist[i] = pulse(i);
      }
   }
  
   SinglePulseShapeFunctor::~SinglePulseShapeFunctor() {
   }
  
   double SinglePulseShapeFunctor::operator()(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i;

      //calculate chisquare
      double chisq = 0;
      double delta;
      std::array<float,nbins> pulse_shape = std::move(funcHPDShape(pars,pulse_hist));
      for (i=0;i<nbins; ++i) {
         delta = (psFit_y[i]- pulse_shape[i])/psFit_erry[i];
         chisq += delta*delta;
      }
      return chisq;
   }
  
   DoublePulseShapeFunctor::DoublePulseShapeFunctor(const HcalPulseShapes::Shape& pulse) {
      for(int i=0;i<256;i++) {
         pulse_hist[i] = pulse(i);
      }
   }

   DoublePulseShapeFunctor::~DoublePulseShapeFunctor() {
   }

   double DoublePulseShapeFunctor::operator()(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i;

      //calculate chisquare
      double chisq = 0;
      double delta;
      //double val[1];
      std::array<float,nbins> pulse_shape = std::move(func_DoublePulse_HPDShape(pars,pulse_hist));
      for (i=0;i<nbins; ++i) {
         delta = (psFit_y[i]- pulse_shape[i])/psFit_erry[i];
         chisq += delta*delta;
      }
      return chisq;
   }

}

PulseShapeFitOOTPileupCorrection::PulseShapeFitOOTPileupCorrection()
{
}

void PulseShapeFitOOTPileupCorrection::apply(const CaloSamples & cs, const std::vector<int> & capidvec, /*const HcalCoder & coder,*/
                       const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const
{
   FitterFuncs::cntNANinfit = 0;

//   CaloSamples cs;
//   coder.adc2fC(digi,cs);
   std::vector<double> chargeVec, pedVec;
   std::vector<double> energyVec, pedenVec;
   double TSTOT = 0, TStrig = 0; // in fC
   double TSTOTen = 0; // in GeV
   for(int ip=0; ip<cs.size(); ip++){
//      const int capid = digi[ip].capid();
      const int capid = capidvec[ip];
      double charge = cs[ip];
      double ped = calibs.pedestal(capid);
      double gain = calibs.respcorrgain(capid);

      double energy = charge*gain;
      double peden = ped*gain;

      chargeVec.push_back(charge); pedVec.push_back(ped);
      energyVec.push_back(energy); pedenVec.push_back(peden);

      TSTOT += charge - ped;
      TSTOTen += energy - peden;
      if( ip ==4 ){
         TStrig = charge - ped;
      }
   }
   std::vector<double> fitParsVec;
   if( TStrig >= 4 && TSTOT >= 10 ){
      pulseShapeFit(energyVec, pedenVec, chargeVec, pedVec, TSTOTen, fitParsVec, spsf_, dpsf_);
//      double time = fitParsVec[1], ampl = fitParsVec[0], uncorr_ampl = fitParsVec[0];
   }
   correctedOutput.swap(fitParsVec); correctedOutput.push_back(FitterFuncs::cntNANinfit);
}

int PulseShapeFitOOTPileupCorrection::pulseShapeFit(const std::vector<double> & energyVec, const std::vector<double> & pedenVec, const std::vector<double> &chargeVec, const std::vector<double> &pedVec, const double TSTOTen, std::vector<double> &fitParsVec, const std::auto_ptr<FitterFuncs::SinglePulseShapeFunctor>& spsf, const std::auto_ptr<FitterFuncs::DoublePulseShapeFunctor>& dpsf) const{

   int n_max=0;
   int n_above_thr=0;
   int first_above_thr_index=-1;
   int max_index[10]={0,0,0,0,0,0,0,0,0,0};

   double TSMAX=0;
   double TSMAX_NOPED=0;
   int i_tsmax=0;

   for(int i=0;i<10;i++){
      if(energyVec[i]>TSMAX){
         TSMAX=energyVec[i];
         TSMAX_NOPED=energyVec[i]-pedenVec[i];
         i_tsmax = i;
      }
   }

   double TIMES[10]={-100,-75,-50,-25,0,25,50,75,100,125};

   if(n_max==0){
      max_index[0]=i_tsmax;
   }

   double error = 1.;
   for(int i=0;i<10;i++){
      FitterFuncs::psFit_x[i]=i;
      FitterFuncs::psFit_y[i]=energyVec[i];
      FitterFuncs::psFit_erry[i]=error;
   }

   TFitterMinuit * gMinuit = new TFitterMinuit();
   gMinuit->SetPrintLevel(-1);

   for(int i=0;i!=10;++i){
      if((chargeVec[i])>6){
         n_above_thr++;
         if(first_above_thr_index==-1){
            first_above_thr_index=i;
         }
      }
   }

   // Fixed Maximum Finder
   for( int i=0 ; i < 10; ++i ) {
      switch( i ) {
         case 0:
            if(chargeVec[i]<=6 && chargeVec[i+1]<=6) continue;
            if( chargeVec[i+1] < chargeVec[i] ) {
               max_index[n_max++] = i;
            }
            break;
         case 9:
            if(chargeVec[i]<=6 && chargeVec[i-1]<=6) continue;
            if( chargeVec[i-1] < chargeVec[i] ) {
               max_index[n_max++] = i;
            }
            break;
         default:
            if(chargeVec[i-1]<=6 && chargeVec[i]<=6 && chargeVec[i+1]<=6) continue;
            if( chargeVec[i-1] < chargeVec[i] && chargeVec[i+1] < chargeVec[i]) {
               max_index[n_max++] = i;
            }
            break;
         }
      }

      if(n_max==0){
         max_index[0]=i_tsmax;
        //n_max=1; // there's still one max if you didn't find any...
      }

      if(n_above_thr<=5){
         FitterFuncs::PulseShapeFCN<FitterFuncs::SinglePulseShapeFunctor>* temp = new FitterFuncs::PulseShapeFCN<FitterFuncs::SinglePulseShapeFunctor>(spsf.get());
         gMinuit->SetMinuitFCN(temp); // bai bai!

         // Set starting values and step sizes for parameters
         double vstart[3] = {TIMES[i_tsmax-1],TSMAX_NOPED,0};

         double step[3] = {0.1,0.1,0.1};
         gMinuit->Clear();
         gMinuit->SetParameter(0, "time", vstart[0], step[0], -100,75);
         gMinuit->SetParameter(1, "energy", vstart[1], step[1], 0,TSTOTen);
         gMinuit->SetParameter(2, "ped", vstart[2], step[2], 0,TSTOTen);
         double chi2=9999.;
         for(int tries=0; tries<=3;tries++){
            // Now ready for minimization step
            gMinuit->CreateMinimizer( TFitterMinuit::kMigrad );
            gMinuit->Minimize();

            double chi2valfit,edm,errdef;
            int nvpar,nparx;
            gMinuit->GetStats(chi2valfit,edm,errdef,nvpar,nparx);

            if(chi2>chi2valfit+0.01) {
               chi2=chi2valfit;
               if(tries==0){
                  gMinuit->CreateMinimizer( TFitterMinuit::kScan );
                  gMinuit->Minimize();
               } else if(tries==1){
                  gMinuit->SetStrategy(1);
               } else if(tries==2){
                  gMinuit->SetStrategy(2);
               }
            } else {
               break;
            }
         }
      } else {
         FitterFuncs::PulseShapeFCN<FitterFuncs::DoublePulseShapeFunctor>* temp = new FitterFuncs::PulseShapeFCN<FitterFuncs::DoublePulseShapeFunctor>(dpsf.get());
         gMinuit->SetMinuitFCN(temp); // bai bai!

         if(n_max==1){
            // Set starting values and step sizes for parameters
            double vstart[5] = {TIMES[i_tsmax-1],TIMES[first_above_thr_index-1],TSMAX_NOPED,0,0};

            Double_t step[5] = {0.1,0.1,0.1,0.1,0.1};
            gMinuit->Clear();
            gMinuit->SetParameter(0, "time1", vstart[0], step[0], -100,75);
            gMinuit->SetParameter(1, "time2", vstart[1], step[1], -100,75);
            gMinuit->SetParameter(2, "energy1", vstart[2], step[2], 0,TSTOTen);
            gMinuit->SetParameter(3, "energy2", vstart[3], step[3], 0,TSTOTen);
            gMinuit->SetParameter(4, "ped", vstart[4], step[4], 0,TSTOTen);

            double chi2=9999.;
            for(int tries=0; tries<=3;tries++) {
            // Now ready for minimization step
               gMinuit->CreateMinimizer( TFitterMinuit::kMigrad );
               gMinuit->Minimize();

               double chi2valfit,edm,errdef;
               int nvpar,nparx;
               gMinuit->GetStats(chi2valfit,edm,errdef,nvpar,nparx);

               if(chi2>chi2valfit+0.01) {
                  chi2=chi2valfit;
                  if(tries==0){
                     gMinuit->CreateMinimizer( TFitterMinuit::kScan );
                     gMinuit->Minimize();
                } else if(tries==1) {
                   gMinuit->SetStrategy(1);
                } else if(tries==2) {
                   gMinuit->SetStrategy(2);
                }
              } else {
                 break;
              }
           }
        } else if(n_max>=2) {
           // Set starting values and step sizes for parameters
           double vstart[5] = {TIMES[max_index[0]-1],TIMES[max_index[1]-1],TSMAX_NOPED,0,0};

           double step[5] = {0.1,0.1,0.1,0.1,0.1};
           gMinuit->Clear();
           gMinuit->SetParameter(0, "time1", vstart[0], step[0], -100,75);
           gMinuit->SetParameter(1, "time2", vstart[1], step[1], -100,75);
           gMinuit->SetParameter(2, "energy1", vstart[2], step[2], 0,TSTOTen);
           gMinuit->SetParameter(3, "energy2", vstart[3], step[3], 0,TSTOTen);
           gMinuit->SetParameter(4, "ped", vstart[4], step[4], 0,TSTOTen);

           double chi2=9999.;
           for(int tries=0; tries<=3;tries++) {
           // Now ready for minimization step
              gMinuit->CreateMinimizer( TFitterMinuit::kMigrad );
              gMinuit->Minimize();

              double chi2valfit,edm,errdef;
              int nvpar,nparx;

              gMinuit->GetStats(chi2valfit,edm,errdef,nvpar,nparx);

              if(chi2>chi2valfit+0.01) {
                 chi2=chi2valfit;
                 if(tries==0){
                    gMinuit->CreateMinimizer( TFitterMinuit::kScan );
                    gMinuit->Minimize();
                 } else if(tries==1) {
                    gMinuit->SetStrategy(1);
                 } else if(tries==2) {
                    gMinuit->SetStrategy(2);
                 }
              } else {
                 break;
              }
           }
        }
     }

     double timeval1fit=-999;
     double chargeval1fit=-999;
     double timeval2fit=-999;
     double chargeval2fit=-999;
     double pedvalfit=-999;

     if(n_above_thr<=5) {
        timeval1fit = gMinuit->GetParameter(0);
        chargeval1fit = gMinuit->GetParameter(1);
        pedvalfit = gMinuit->GetParameter(2);
     } else {
        timeval1fit = gMinuit->GetParameter(0);
        timeval2fit = gMinuit->GetParameter(1);
        chargeval1fit = gMinuit->GetParameter(2);
        chargeval2fit = gMinuit->GetParameter(3);
        pedvalfit = gMinuit->GetParameter(4);
     }

     double chi2valfit,edm,errdef;
     int nvpar,nparx;
     int fitStatus = gMinuit->GetStats(chi2valfit,edm,errdef,nvpar,nparx);

     double timevalfit=0.;
     double chargevalfit=0.;
     if(n_above_thr<=5) {
        timevalfit=timeval1fit;
        chargevalfit=chargeval1fit;
     } else {
        if(fabs(timeval1fit)<fabs(timeval2fit)) {// if timeval1fit and timeval2fit are differnt, choose the one which is closer to zero
           timevalfit=timeval1fit;
           chargevalfit=chargeval1fit;
        } else if(fabs(timeval2fit)<fabs(timeval1fit)) {// if timeval1fit and timeval2fit are differnt, choose the one which is closer to zero
           timevalfit=timeval2fit;
           chargevalfit=chargeval2fit;
        } else if(timeval1fit==timeval2fit) { // if the two times are the same, then for charge we just sum the two  
           timevalfit=(timeval1fit+timeval2fit)/2;
           chargevalfit=chargeval1fit+chargeval2fit;
        } else {
           timevalfit=-999.;
           chargevalfit=-999.;
        }
     }

     if( gMinuit ) delete gMinuit;

     fitParsVec.clear();

     fitParsVec.push_back(chargevalfit);
     fitParsVec.push_back(timevalfit);
     fitParsVec.push_back(pedvalfit);
     fitParsVec.push_back(chi2valfit);

     return fitStatus;
}
