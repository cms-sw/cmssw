//-*-c++-*-
//-*-Event.cpp-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////

#include "GeneratorInterface/ExhumeInterface/interface/Event.h"
#include "CLHEP/Random/RandomEngine.h"
//////////////////////////////////////////////////////////////////////////////
Exhume::Event::Event(CrossSection &Process_, CLHEP::HepRandomEngine *engine) {
  Process = &Process_;
  Setx1Max(1.0);
  Setx2Max(1.0);
  Sett1Max(0.0);
  Sett2Max(0.0);
  Sett1Min(-10.0);
  Sett2Min(-10.0);
  SetMassRange(100, 130);
  NumberOfEvents = 0;
  TotalAttempts = 0;
  ymax = 0.0;
  ymin = 0.0;
  CSMass = 0.0;
  Sigmai = 0.0;
  yRange = 0.0;
  TwoPI = 2.0 * M_PI;
  B = Process->GetB();
  InvB = 1.0 / B;
  InvBlnB = InvB * log(B);
  Root_s = Process->GetRoot_s();
  InvRoot_s = 1.0 / Root_s;
  randomEngine = engine;
  std::cout << "   ++ Random number seed   " << randomEngine->getSeed() << " ++" << std::endl << std::endl;
  Process->SetRandomEngine(engine);
  //srand(Seed_);
}
///////////////////////////////////////////////////////////////////////////////
Exhume::Event::~Event() {}
///////////////////////////////////////////////////////////////////////////////
void Exhume::Event::Generate() {
  bool pass = false;
  do {
    SelectValues();
    Process->SetKinematics(SqrtsHat, Eta, t1, t2, Phi1, Phi2);
    Process->SetSubParameters();
    Process->SetPartons();
    CSi = Process->Differential();

    wgt = GetFunc(SqrtsHat);
    CSMass += wgt;
    double ebt = exp(B * (t1 + t2 - t1Max - t2Max)) * Process->SubParameterWeight();
    wgt = 1.1 * wgt * ebt;

    if (CSi > wgt) {  //Don't want this to happen very often - it is slow.
                      //Choose your cuts sensibly!
      Process->MaximiseSubParameters();
      wgt = WeightFunc(SqrtsHat);
      AddPoint(SqrtsHat, wgt);
      wgt = 1.1 * wgt * ebt;
      Process->SetKinematics(SqrtsHat, Eta, t1, t2, Phi1, Phi2);
      Process->SetSubParameters();
    }

    CSi = CSi / wgt;

    Sigmai += CSi;

    if (CSi > VonNeu) {
      if (CSi > 1.0) {  //Then it has cocked it up completely

        std::map<double, double> FuncMap_ = GetFuncMap();
        std::map<double, double> LineShape_ = GetLineShape();

        FILE *dat = fopen("FuncMap.dat", "w");

        for (std::map<double, double>::iterator kk = FuncMap_.begin(); kk != FuncMap_.end(); kk++) {
          //fprintf(dat,"%le\t%le\n",kk->first, kk->second);
          fprintf(dat, "%e\t%e\n", kk->first, kk->second);
        }

        fclose(dat);

        dat = fopen("LineShape.dat", "w");

        for (std::map<double, double>::iterator kk = LineShape_.begin(); kk != LineShape_.end(); kk++) {
          //fprintf(dat, "%le\t%le\n",kk->first, kk->second);
          fprintf(dat, "%e\t%e\n", kk->first, kk->second);
        }
        fclose(dat);

        std::cout << "   This should never happen" << std::endl;
        std::cout << "   m   = " << SqrtsHat << std::endl;
        std::cout << "   y   = " << Eta << std::endl;
        std::cout << "   t1  = " << t1 << std::endl;
        std::cout << "   t2  = " << t2 << std::endl;
        std::cout << "   x1  = " << Process->Getx1() << std::endl;
        std::cout << "   x2  = " << Process->Getx2() << std::endl;
        std::cout << "   CSi = " << CSi * wgt << std::endl;
        std::cout << "   wgt = " << wgt << std::endl;
        std::cout << "   Please e-mail the authors." << std::endl
                  << "   Include the files FuncMap.dat and LineShape.dat" << std::endl;
        exit(0);
      }
      pass = true;
      NumberOfEvents++;
      TotalAttempts++;

      Var.push_back(std::pair<double, double>(TotalAttempts * TotalIntegral, Sigmai));

    } else {
      TotalAttempts++;
    }
  } while (!pass);
}
//////////////////////////////////////////////////////////////////////////////
void Exhume::Event::SelectValues() {
  double MRand = randomEngine->flat();

  SqrtsHat = GetValue(MRand);

  double tt1 = randomEngine->flat() * (tt1max - tt1min) + tt1min;
  double tt2 = randomEngine->flat() * (tt2max - tt2min) + tt2min;

  t1 = InvB * log(tt1) + InvBlnB;
  t2 = InvB * log(tt2) + InvBlnB;

  Phi1 = TwoPI * randomEngine->flat();
  Phi2 = TwoPI * randomEngine->flat();

  // y = 0.5 log(x1/x2) and at t1 = t2 = 0 x1x2 = sHat/s
  // so the max kinematically allowed value of x1 is
  // x1max = sHat/ s / x2min
  // x2max =  sHat/ s / x1min and so
  // ymax = 0.5 log(x1max / x2min) = 0.5 log(x1max^2 * s/sHat)
  // ymin = 0.5 log(x1min / x2max) = -0.5 log (x2max^2 * s/sHat)

  double yymax = log(x1Max * Root_s / SqrtsHat);
  double yymin = log(InvRoot_s * SqrtsHat / x2Max);

  yRange += (yymax - yymin);

  Eta = (yymax - yymin) * randomEngine->flat() + yymin;

  VonNeu = randomEngine->flat();

  return;
}
//////////////////////////////////////////////////////////////////////////////
void Exhume::Event::SetParameterSpace() {
  tt1max = InvB * exp(t1Max * B);
  tt2max = InvB * exp(t2Max * B);
  tt1min = InvB * exp(t1Min * B);
  tt2min = InvB * exp(t2Min * B);

  Process->MaximiseSubParameters();

  WeightInit(MinMass, MaxMass);

  return;
}
//////////////////////////////////////////////////////////////////////////////
double Exhume::Event::WeightFunc(const double &mm_) {
  Process->SetKinematics(mm_, 0.0, t1Max, t2Max, 0.0, 0.0);

  if (mm_ < 65.0) {
    return (1.00025 * Process->Differential());  //Factor 1.00025 accounts for
                                                 //slight kink in lumi function
  }

  return (Process->Differential());
}
//////////////////////////////////////////////////////////////////////////////
double Exhume::Event::CrossSectionCalculation() {
  double InvTotalAttempts = 1.0 / TotalAttempts;
  double yy = yRange * InvTotalAttempts;

  //\phi dependence is already integrated out of the differential luminosity

  return (1.1 * TotalIntegral * (tt1max - tt1min) * (tt2max - tt2min) * yy * fabs(Process->SubParameterRange()) *
          Sigmai * InvTotalAttempts);
}
//////////////////////////////////////////////////////////////////////////////
