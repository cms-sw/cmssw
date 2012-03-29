#include "../interface/ProcessNormalization.h"

#include <cmath>
#include <cassert>
#include <cstdio>

ProcessNormalization::ProcessNormalization(const char *name, const char *title, double nominal) :
        RooAbsReal(name,title),
        nominalValue_(nominal),
        thetaList_("thetaList","List of nuisances for symmetric kappas", this), 
        asymmThetaList_("asymmThetaList","List of nuisances for asymmetric kappas", this), 
        otherFactorList_("otherFactorList","Other multiplicative terms", this)
{ 
}

ProcessNormalization::ProcessNormalization(const char *name, const char *title, RooAbsReal &nominal) :
        RooAbsReal(name,title),
        nominalValue_(1.0),
        thetaList_("thetaList", "List of nuisances for symmetric kappas", this), 
        asymmThetaList_("asymmThetaList", "List of nuisances for asymmetric kappas", this), 
        otherFactorList_("otherFactorList", "Other multiplicative terms", this)
{ 
    otherFactorList_.add(nominal);
}

ProcessNormalization::ProcessNormalization(const ProcessNormalization &other, const char *newname) :
        RooAbsReal(other, newname ? newname : other.GetName()),
        nominalValue_(other.nominalValue_),
        logKappa_(other.logKappa_),
        thetaList_("thetaList", this, other.thetaList_), 
        logAsymmKappa_(other.logAsymmKappa_),
        asymmThetaList_("asymmThetaList", this, other.asymmThetaList_), 
        otherFactorList_("otherFactorList", this, other.otherFactorList_)
{
}

ProcessNormalization::~ProcessNormalization() {}

void ProcessNormalization::addLogNormal(double kappa, RooAbsReal &theta) {
    if (kappa != 0.0 && kappa != 1.0) {
        logKappa_.push_back(std::log(kappa));
        thetaList_.add(theta);
    }
}

void ProcessNormalization::addAsymmLogNormal(double kappaLo, double kappaHi, RooAbsReal &theta) {
    if (fabs(kappaLo*kappaHi - 1) < 1e-5) {
        addLogNormal(kappaHi, theta);
    } else {
        logAsymmKappa_.push_back(std::make_pair(std::log(kappaLo), std::log(kappaHi)));
        asymmThetaList_.add(theta);
    }
}

void ProcessNormalization::addOtherFactor(RooAbsReal &factor) {
    otherFactorList_.add(factor);
}

Double_t ProcessNormalization::evaluate() const {
    double logVal = 0.0;
    if (!logKappa_.empty()) {
        RooLinkedListIter iterTheta = thetaList_.iterator();
        std::vector<double>::const_iterator logKappa = logKappa_.begin();
        for (RooAbsReal *theta = (RooAbsReal*) iterTheta.Next(); theta != 0; theta = (RooAbsReal*) iterTheta.Next(), ++logKappa) {
            logVal += theta->getVal() * (*logKappa);
        }
    }
    if (!logAsymmKappa_.empty()) {
        RooLinkedListIter iterTheta = asymmThetaList_.iterator();
        std::vector<std::pair<double,double> >::const_iterator logKappas = logAsymmKappa_.begin();
        for (RooAbsReal *theta = (RooAbsReal*) iterTheta.Next(); theta != 0; theta = (RooAbsReal*) iterTheta.Next(), ++logKappas) {
            double x = theta->getVal();
            logVal +=  x * logKappaForX(x, *logKappas);
        }
    }
    double norm = nominalValue_;
    if (logVal) norm *= std::exp(logVal);
    if (otherFactorList_.getSize()) {
        RooLinkedListIter iterOther = otherFactorList_.iterator();
        for (RooAbsReal *fact = (RooAbsReal*) iterOther.Next(); fact != 0; fact = (RooAbsReal*) iterOther.Next()) {
            norm *= fact->getVal();
        }
    }
    return norm;
}

Double_t ProcessNormalization::logKappaForX(double x, const std::pair<double,double> &logKappas) const {
    if (fabs(x) >= 0.5) return (x >= 0 ? logKappas.second : -logKappas.first);
    // interpolate between log(kappaHigh) and -log(kappaLow) 
    //    logKappa(x) = avg + halfdiff * h(2x)
    // where h(x) is the 3th order polynomial
    //    h(x) = (3 x^5 - 10 x^3 + 15 x)/8;
    // chosen so that h(x) satisfies the following:
    //      h (+/-1) = +/-1 
    //      h'(+/-1) = 0
    //      h"(+/-1) = 0
    double logKhi =  logKappas.second;
    double logKlo = -logKappas.first;
    double avg = 0.5*(logKhi + logKlo), halfdiff = 0.5*(logKhi - logKlo);
    double twox = x+x, twox2 = twox*twox;
    double alpha = 0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.);
    double ret = avg + alpha*halfdiff;
    return ret;
} 

void ProcessNormalization::dump() const {
    std::cout << "Dumping ProcessNormalization " << GetName() << " @ " << (void*)this << std::endl;
    std::cout << "\tnominal value: " << nominalValue_ << std::endl;
    std::cout << "\tlog-normals (" << logKappa_.size() << "):"  << std::endl;
    for (unsigned int i = 0; i < logKappa_.size(); ++i) {
        std::cout << "\t\t kappa = " << exp(logKappa_[i]) << ", logKappa = " << logKappa_[i] << 
                     ", theta = " << thetaList_.at(i)->GetName() << " = " << ((RooAbsReal*)thetaList_.at(i))->getVal() << std::endl;
    }
    std::cout << "\tasymm log-normals (" << logAsymmKappa_.size() << "):"  << std::endl;
    for (unsigned int i = 0; i < logAsymmKappa_.size(); ++i) {
        std::cout << "\t\t kappaLo = " << exp(logAsymmKappa_[i].first) << ", logKappaLo = " << logAsymmKappa_[i].first << 
                     ", kappaHi = " << exp(logAsymmKappa_[i].second) << ", logKappaHi = " << logAsymmKappa_[i].second << 
                     ", theta = " << thetaList_.at(i)->GetName() << " = " << ((RooAbsReal*)thetaList_.at(i))->getVal() << std::endl;
    }
    std::cout << "\tother terms (" << otherFactorList_.getSize() << "):"  << std::endl;
    for (int i = 0; i < otherFactorList_.getSize(); ++i) {  
        std::cout << "\t\t term " << otherFactorList_.at(i)->GetName() <<
                     " (class " << otherFactorList_.at(i)->ClassName() << 
                     "), value = " << ((RooAbsReal*)otherFactorList_.at(i))->getVal() << std::endl;
    }
    std::cout << std::endl;
}
ClassImp(ProcessNormalization)
