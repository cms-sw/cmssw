#include <climits>
#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"

namespace FitterFuncs{

   int cntNANinfit;
   double psFit_x[10], psFit_y[10], psFit_erry[10]; 

   std::array<float,10> funcHPDShape(const std::vector<double>& pars,
                                     const std::array<float,256>& h1_single, const std::vector<float> &acc25nsVec, const std::vector<float> &diff25nsItvlVec, const std::vector<float> &accVarLenIdxZEROVec, const std::vector<float> &diffVarItvlIdxZEROVec, const std::vector<float> &accVarLenIdxMinusOneVec, const std::vector<float>&diffVarItvlIdxMinusOneVec) {
    // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
      constexpr int ns_per_bx = 25;
      constexpr int num_ns = 250;
      constexpr int num_bx = num_ns/ns_per_bx;

    // zeroing output binned pulse shape
      std::array<float,num_bx> ntmpbin{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };

      int i_start = ( -98.5f - pars[0] >0 ? 0 : (int)fabs(-98.5f-pars[0]) + 1);
      double offset_start = i_start - 98.5f - pars[0];
      if( offset_start == 1.0 ){ offset_start = 0.; i_start-=1; }
      const int bin_start = (int) offset_start;
      const int bin_0_start = ( offset_start < bin_start + 0.5 ? bin_start -1 : bin_start );
      const int iTS_start = i_start/ns_per_bx;
      const int distTo25ns_start = 24 - i_start%ns_per_bx;
      const double factor = offset_start - bin_0_start - 0.5;

      if( offset_start != offset_start){
         cntNANinfit ++;
      }else{
         ntmpbin[iTS_start] = (bin_0_start == -1 ? accVarLenIdxMinusOneVec[distTo25ns_start] + factor * diffVarItvlIdxMinusOneVec[distTo25ns_start]
                                                 : accVarLenIdxZEROVec[distTo25ns_start] + factor * diffVarItvlIdxZEROVec[distTo25ns_start]);
         for(int iTS = iTS_start+1; iTS < num_bx; ++iTS){
            int bin_idx = distTo25ns_start + 1 + (iTS-iTS_start-1)*ns_per_bx + bin_0_start;
            ntmpbin[iTS] = acc25nsVec[bin_idx] + factor * diff25nsItvlVec[bin_idx];
         }
      }

    // now we use ntmpbin to record the final pulse shape
      for(int i=0; i < num_bx; ++i) {
         ntmpbin[i] *= pars[1];
         ntmpbin[i] += pars[2];
      }
      return ntmpbin;
   }

   std::array<float,10> func_DoublePulse_HPDShape(const std::vector<double>& pars,
                                                  const std::array<float,256>& h1_double, const std::vector<float> &acc25nsVec, const std::vector<float> &diff25nsItvlVec, const std::vector<float> &accVarLenIdxZEROVec, const std::vector<float> &diffVarItvlIdxZEROVec, const std::vector<float> &accVarLenIdxMinusOneVec, const std::vector<float>&diffVarItvlIdxMinusOneVec) {
    // pulse shape components over a range of time 0 ns to 255 ns in 1 ns steps
      constexpr int ns_per_bx = 25;
      constexpr int num_ns = 250;
      constexpr int num_bx = num_ns/ns_per_bx;

    // zeroing output binned pulse shape
      std::array<float,num_bx> ntmpbin{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };
      std::array<float,num_bx> ntmpbin2{ {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f} };

      int i_start = ( -98.5f - pars[0] >0 ? 0 : (int)fabs(-98.5f-pars[0]) + 1);
      double offset_start = i_start - 98.5f - pars[0];
      if( offset_start == 1.0 ){ offset_start = 0.; i_start-=1; }
      const int bin_start = (int) offset_start;
      const int bin_0_start = ( offset_start < bin_start + 0.5 ? bin_start -1 : bin_start );
      const int iTS_start = i_start/ns_per_bx;
      const int distTo25ns_start = 24 - i_start%ns_per_bx;
      const double factor = offset_start - bin_0_start - 0.5;

      if( offset_start != offset_start){
         cntNANinfit ++;
      }else{
         ntmpbin[iTS_start] = (bin_0_start == -1 ? accVarLenIdxMinusOneVec[distTo25ns_start] + factor * diffVarItvlIdxMinusOneVec[distTo25ns_start]
                                                 : accVarLenIdxZEROVec[distTo25ns_start] + factor * diffVarItvlIdxZEROVec[distTo25ns_start]);

         for(int iTS = iTS_start+1; iTS < num_bx; ++iTS){
            int bin_idx = distTo25ns_start + 1 + (iTS-iTS_start-1)*ns_per_bx + bin_0_start;
            ntmpbin[iTS] = acc25nsVec[bin_idx] + factor * diff25nsItvlVec[bin_idx];
         }
      }

      int i_start2 = ( -98.5f - pars[1] >0 ? 0 : (int)fabs(-98.5f-pars[1]) + 1);
      double offset_start2 = i_start2 - 98.5f - pars[1];
      if( offset_start2 == 1.0 ){ offset_start2 = 0.; i_start2 -=1; }
      const int bin_start2 = (int) offset_start2;
      const int bin_0_start2 = ( offset_start2 < bin_start2 + 0.5 ? bin_start2 -1 : bin_start2 );
      const int iTS_start2 = i_start2/ns_per_bx;
      const int distTo25ns_start2 = 24 - i_start2%ns_per_bx;
      const double factor2 = offset_start2 - bin_0_start2 - 0.5;

      if( offset_start2 != offset_start2){
         cntNANinfit ++;
      }else{
         ntmpbin2[iTS_start2] = (bin_0_start2 == -1 ? accVarLenIdxMinusOneVec[distTo25ns_start2] + factor2 * diffVarItvlIdxMinusOneVec[distTo25ns_start2]
                                                    : accVarLenIdxZEROVec[distTo25ns_start2] + factor2 * diffVarItvlIdxZEROVec[distTo25ns_start2]);

         for(int iTS = iTS_start2+1; iTS < num_bx; ++iTS){
            int bin_idx2 = distTo25ns_start2 + 1 + (iTS-iTS_start2-1)*ns_per_bx + bin_0_start2;
            ntmpbin2[iTS] = acc25nsVec[bin_idx2] + factor2 * diff25nsItvlVec[bin_idx2];
         }
      }

    // now we use ntmpbin to record the final pulse shape
      for(int i=0; i < num_bx; ++i) {
         ntmpbin[i] *= pars[2];
         ntmpbin2[i] *= pars[3];
         ntmpbin[i] += ntmpbin2[i] + pars[4];
      }
      return ntmpbin;
   }

   PulseShapeFunctor::PulseShapeFunctor(const HcalPulseShapes::Shape& pulse) : 
      acc25nsVec(256), diff25nsItvlVec(256),
      accVarLenIdxZEROVec(25), diffVarItvlIdxZEROVec(25), 
      accVarLenIdxMinusOneVec(25), diffVarItvlIdxMinusOneVec(25) {

      for(int i=0;i<256;++i) {
         pulse_hist[i] = pulse(i);
      }
// Accumulate 25ns for each starting point of 0, 1, 2, 3...
      for(int i=0; i<256; ++i){
         for(int j=i; j<i+25; ++j){
            acc25nsVec[i] += ( j < 256? pulse_hist[j] : pulse_hist[255]);
         }
         diff25nsItvlVec[i] = ( i+25 < 256? pulse_hist[i+25] - pulse_hist[i] : pulse_hist[255] - pulse_hist[i]);
      }
// Accumulate different ns for starting point of index either 0 or -1
      for(int i=0; i<25; ++i){
         if( i==0 ){
            accVarLenIdxZEROVec[0] = pulse_hist[0];
            accVarLenIdxMinusOneVec[i] = pulse_hist[0];
         } else{
            accVarLenIdxZEROVec[i] = accVarLenIdxZEROVec[i-1] + pulse_hist[i];
            accVarLenIdxMinusOneVec[i] = accVarLenIdxMinusOneVec[i-1] + pulse_hist[i-1];
         }
         diffVarItvlIdxZEROVec[i] = pulse_hist[i+1] - pulse_hist[0];
         diffVarItvlIdxMinusOneVec[i] = pulse_hist[i] - pulse_hist[0];
      }
   }
  
   PulseShapeFunctor::~PulseShapeFunctor() {
   }
  
   double PulseShapeFunctor::EvalSinglePulse(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i =0;

      //calculate chisquare
      double chisq = 0;
      double delta =0;
      std::array<float,nbins> pulse_shape = std::move(funcHPDShape(pars,pulse_hist, acc25nsVec, diff25nsItvlVec, accVarLenIdxZEROVec, diffVarItvlIdxZEROVec, accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));
      for (i=0;i<nbins; ++i) {
         delta = (psFit_y[i]- pulse_shape[i])/psFit_erry[i];
         chisq += delta*delta;
      }
      return chisq;
   }

   double PulseShapeFunctor::EvalDoublePulse(const std::vector<double>& pars) const {
      constexpr unsigned nbins = 10;
      unsigned i =0;

      //calculate chisquare
      double chisq = 0;
      double delta = 0;
      //double val[1];
      std::array<float,nbins> pulse_shape = std::move(func_DoublePulse_HPDShape(pars,pulse_hist, acc25nsVec, diff25nsItvlVec, accVarLenIdxZEROVec, diffVarItvlIdxZEROVec, accVarLenIdxMinusOneVec, diffVarItvlIdxMinusOneVec));
      for (i=0;i<nbins; ++i) {
         delta = (psFit_y[i]- pulse_shape[i])/psFit_erry[i];
         chisq += delta*delta;
      }
      return chisq;
   }
 
   std::auto_ptr<PulseShapeFunctor> psfPtr_;

   double singlePulseShapeFunc( const double *x ) {
      std::vector<double> pars(x, x+3);
      return psfPtr_->EvalSinglePulse(pars);
   }

   double doublePulseShapeFunc( const double *x ) {
      std::vector<double> pars(x, x+5);
      return psfPtr_->EvalDoublePulse(pars);
   }
}

PulseShapeFitOOTPileupCorrection::PulseShapeFitOOTPileupCorrection() : cntsetPulseShape(0), chargeThreshold_(6.)
{
   hybridfitter = new PSFitter::HybridMinimizer(PSFitter::HybridMinimizer::kMigrad);
   iniTimesArr = { {-100,-75,-50,-25,0,25,50,75,100,125} };
}

PulseShapeFitOOTPileupCorrection::~PulseShapeFitOOTPileupCorrection()
{ 
   if(hybridfitter) delete hybridfitter;
}

void PulseShapeFitOOTPileupCorrection::setPulseShapeTemplate(const HcalPulseShapes::Shape& ps) {
   if( cntsetPulseShape ) return;
   ++ cntsetPulseShape;
   FitterFuncs::psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps));
}

void PulseShapeFitOOTPileupCorrection::resetPulseShapeTemplate(const HcalPulseShapes::Shape& ps) {
   ++ cntsetPulseShape;
   FitterFuncs::psfPtr_.reset(new FitterFuncs::PulseShapeFunctor(ps));
}

void PulseShapeFitOOTPileupCorrection::apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const
{
   FitterFuncs::cntNANinfit = 0;

   const unsigned int cssize = cs.size();
   double chargeArr[cssize], pedArr[cssize];
   double energyArr[cssize], pedenArr[cssize];
   double tsTOT = 0, tstrig = 0; // in fC
   double tsTOTen = 0; // in GeV
   for(unsigned int ip=0; ip<cssize; ++ip){
      const int capid = capidvec[ip];
      double charge = cs[ip];
      double ped = calibs.pedestal(capid);
      double gain = calibs.respcorrgain(capid);

      double energy = charge*gain;
      double peden = ped*gain;

      chargeArr[ip] = charge; pedArr[ip] = ped;
      energyArr[ip] = energy; pedenArr[ip] = peden;

      tsTOT += charge - ped;
      tsTOTen += energy - peden;
      if( ip ==4 ){
         tstrig = charge - ped;
      }
   }
   std::vector<double> fitParsVec;
   if( tstrig >= 4 && tsTOT >= 10 ){
      pulseShapeFit(energyArr, pedenArr, chargeArr, pedArr, tsTOTen, fitParsVec);
//      double time = fitParsVec[1], ampl = fitParsVec[0], uncorr_ampl = fitParsVec[0];
   }
   correctedOutput.swap(fitParsVec); correctedOutput.push_back(FitterFuncs::cntNANinfit);
}

constexpr char const* sp_varNames[] = {"time", "energy", "ped"};
constexpr char const* dp_varNames[] = {"time1", "time2", "energy1", "energy2", "ped"};

int PulseShapeFitOOTPileupCorrection::pulseShapeFit(const double * energyArr, const double * pedenArr, const double *chargeArr, const double *pedArr, const double tsTOTen, std::vector<double> &fitParsVec) const{

   int n_max=0;
   int n_above_thr=0;
   int first_above_thr_index=-1;
   int max_index[10]={0,0,0,0,0,0,0,0,0,0};

   double tsMAX=0;
   double tsMAX_NOPED=0;
   int i_tsmax=0;

   for(int i=0;i<10;++i){
      if(energyArr[i]>tsMAX){
         tsMAX=energyArr[i];
         tsMAX_NOPED=energyArr[i]-pedenArr[i];
         i_tsmax = i;
      }
   }

   if(n_max==0){
      max_index[0]=i_tsmax;
   }

   double error = 1.;
   for(int i=0;i<10;++i){
      FitterFuncs::psFit_x[i]=i;
      FitterFuncs::psFit_y[i]=energyArr[i];
      FitterFuncs::psFit_erry[i]=error;
   }

   for(int i=0;i!=10;++i){
      if((chargeArr[i])>chargeThreshold_){
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
            if(chargeArr[i]<=chargeThreshold_ && chargeArr[i+1]<=chargeThreshold_) continue;
            if( chargeArr[i+1] < chargeArr[i] ) {
               max_index[n_max++] = i;
            }
            break;
         case 9:
            if(chargeArr[i]<=chargeThreshold_ && chargeArr[i-1]<=chargeThreshold_) continue;
            if( chargeArr[i-1] < chargeArr[i] ) {
               max_index[n_max++] = i;
            }
            break;
         default:
            if(chargeArr[i-1]<=chargeThreshold_ && chargeArr[i]<=chargeThreshold_ && chargeArr[i+1]<=chargeThreshold_) continue;
            if( chargeArr[i-1] < chargeArr[i] && chargeArr[i+1] < chargeArr[i]) {
               max_index[n_max++] = i;
            }
            break;
         }
      }

      if(n_max==0){
         max_index[0]=i_tsmax;
        //n_max=1; // there's still one max if you didn't find any...
      }

      bool fitStatus = false;
      if(n_above_thr<=5){
         // Set starting values and step sizes for parameters
         double vstart[3] = {iniTimesArr[i_tsmax], tsMAX_NOPED, 0};
         double step[3] = {0.1, 0.1, 0.1};

         ROOT::Math::Functor spfunctor(&FitterFuncs::singlePulseShapeFunc, 3);

         hybridfitter->SetFunction(spfunctor);
         hybridfitter->Clear();
         hybridfitter->SetLimitedVariable(0, sp_varNames[0], vstart[0], step[0], -100, 75);
         hybridfitter->SetLimitedVariable(1, sp_varNames[1], vstart[1], step[1], 0, tsTOTen);
         hybridfitter->SetLimitedVariable(2, sp_varNames[2], vstart[2], step[2], 0, tsTOTen);

         double chi2=9999.;
         for(int tries=0; tries<=3;++tries){

            hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kMigrad);
            fitStatus = hybridfitter->Minimize();
            double chi2valfit = hybridfitter->MinValue();

            if(chi2>chi2valfit+0.01) {
               chi2=chi2valfit;
               if(tries==0){
                 hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kScan);
                 hybridfitter->Minimize();
               } else if(tries==1){
                  hybridfitter->SetStrategy(1);
               } else if(tries==2){
                  hybridfitter->SetStrategy(2);
               }
            } else {
               break;
            }
         }

      } else {

         ROOT::Math::Functor dpfunctor(&FitterFuncs::doublePulseShapeFunc, 5);

         hybridfitter->SetFunction(dpfunctor);

         if(n_max==1){
            // Set starting values and step sizes for parameters
            double vstart[5] = {iniTimesArr[i_tsmax], iniTimesArr[first_above_thr_index], tsMAX_NOPED, 0, 0};
            Double_t step[5] = {0.1, 0.1, 0.1, 0.1, 0.1};

            hybridfitter->Clear();
            hybridfitter->SetLimitedVariable(0, dp_varNames[0], vstart[0], step[0], -100, 75);
            hybridfitter->SetLimitedVariable(1, dp_varNames[1], vstart[1], step[1], -100, 75);
            hybridfitter->SetLimitedVariable(2, dp_varNames[2], vstart[2], step[2], 0, tsTOTen);
            hybridfitter->SetLimitedVariable(3, dp_varNames[3], vstart[3], step[3], 0, tsTOTen);
            hybridfitter->SetLimitedVariable(4, dp_varNames[4], vstart[4], step[4], 0, tsTOTen);

            double chi2=9999.;
            for(int tries=0; tries<=3;++tries) {
            // Now ready for minimization step

               hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kMigrad);
               fitStatus = hybridfitter->Minimize(); 
               double chi2valfit = hybridfitter->MinValue();

               if(chi2>chi2valfit+0.01) {
                  chi2=chi2valfit;
                if(tries==0){
                  hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kScan);
                  hybridfitter->Minimize();
                } else if(tries==1) {
                   hybridfitter->SetStrategy(1);
                } else if(tries==2) {
                   hybridfitter->SetStrategy(2);
                }
              } else {
                 break;
              }
           }
        } else if(n_max>=2) {
           // Set starting values and step sizes for parameters
           double vstart[5] = {iniTimesArr[max_index[0]], iniTimesArr[max_index[1]], tsMAX_NOPED, 0, 0};
           double step[5] = {0.1, 0.1, 0.1, 0.1, 0.1};

           hybridfitter->Clear();
           hybridfitter->SetLimitedVariable(0, dp_varNames[0], vstart[0], step[0], -100, 75);
           hybridfitter->SetLimitedVariable(1, dp_varNames[1], vstart[1], step[1], -100, 75);
           hybridfitter->SetLimitedVariable(2, dp_varNames[2], vstart[2], step[2], 0, tsTOTen);
           hybridfitter->SetLimitedVariable(3, dp_varNames[3], vstart[3], step[3], 0, tsTOTen);
           hybridfitter->SetLimitedVariable(4, dp_varNames[4], vstart[4], step[4], 0, tsTOTen);

           double chi2=9999.;
           for(int tries=0; tries<=3;++tries) {
           // Now ready for minimization step

              hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kMigrad);
              fitStatus = hybridfitter->Minimize();
              double chi2valfit = hybridfitter->MinValue();

              if(chi2>chi2valfit+0.01) {
                 chi2=chi2valfit;
                 if(tries==0){
                   hybridfitter->SetMinimizerType(PSFitter::HybridMinimizer::kScan);
                   hybridfitter->Minimize();
                 } else if(tries==1) {
                    hybridfitter->SetStrategy(1);
                 } else if(tries==2) {
                    hybridfitter->SetStrategy(2);
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
        timeval1fit = hybridfitter->X()[0];
        chargeval1fit = hybridfitter->X()[1];
        pedvalfit = hybridfitter->X()[2];
     } else {
        timeval1fit = hybridfitter->X()[0];
        timeval2fit = hybridfitter->X()[1];
        chargeval1fit = hybridfitter->X()[2];
        chargeval2fit = hybridfitter->X()[3];
        pedvalfit = hybridfitter->X()[4];
     }

     int outfitStatus = (fitStatus ? 1: 0 );
     double chi2valfit = hybridfitter->MinValue();
     
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

     fitParsVec.clear();

     fitParsVec.push_back(chargevalfit);
     fitParsVec.push_back(timevalfit);
     fitParsVec.push_back(pedvalfit);
     fitParsVec.push_back(chi2valfit);

     return outfitStatus;
}
