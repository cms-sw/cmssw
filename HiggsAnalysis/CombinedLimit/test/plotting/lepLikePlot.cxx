using namespace RooFit;
using namespace RooStats;
#include <algorithm>
#include <vector>

RooStats::HypoTestResult *readLepFile(TDirectory *toyDir,  double rValue) {
    TString prefix = TString::Format("HypoTestResult_r%g_",rValue);
    RooStats::HypoTestResult *ret = 0;
    TIter next(toyDir->GetListOfKeys()); 
    TKey *k;
    while ((k = (TKey *) next()) != 0) {
        if (TString(k->GetName()).Index(prefix) != 0) continue;
        RooStats::HypoTestResult *toy = (RooStats::HypoTestResult *)(toyDir->Get(k->GetName()));
        if (toy == 0) continue;
        if (ret == 0) {
            ret = new RooStats::HypoTestResult(*toy);
        } else {
            ret->Append(toy);
        }
    }
    return ret;
}

struct LepBand { 
    int n;
    TGraphAsymmErrors *obs, *exp68, *exp95;
};

void addLepPoint(LepBand &b, double x, TString file, double r=1.0) {
    if (gSystem->AccessPathName(file)) return;
    TFile *fIn = TFile::Open(file); 
    if (fIn == 0) { std::cerr << "Cannot open " << file << std::endl; return; }
    TDirectory *toyDir = fIn->GetDirectory("toys");
    if (!toyDir) throw std::logic_error("Cannot use readHypoTestResult: option toysFile not specified, or input file empty");
    RooStats::HypoTestResult *res = readLepFile(toyDir, r);
    double clsObs = res->CLs(), clsObsErr = res->CLsError();
    double clsbObs = res->CLsplusb(), clsbObsErr = res->CLsplusbError();
    //std::cout << "Observed CLs  = " << res->CLs() << " +/- " << res->CLsError() << std::endl;    
    //std::cout << "Observed CLsb = " << res->CLsplusb() << " +/- " << res->CLsplusbError() << std::endl;    
    std::vector<double> samples = res->GetNullDistribution()->GetSamplingDistribution();
    int nd = samples.size();
    std::sort(samples.begin(), samples.end());
    double median = (samples.size() % 2 == 0 ? 0.5*(samples[nd/2]+samples[nd/2+1]) : samples[nd/2]);
    double summer68 = samples[floor(nd * 0.5*(1-0.68)+0.5)], winter68 =  samples[TMath::Min(int(floor(nd * 0.5*(1+0.68)+0.5)), nd-1)];
    double summer95 = samples[floor(nd * 0.5*(1-0.95)+0.5)], winter95 =  samples[TMath::Min(int(floor(nd * 0.5*(1+0.95)+0.5)), nd-1)];
    res->SetTestStatisticData(median+1e-6);    double clsMid = res->CLs(), clsbMid = res->CLsplusb();
    res->SetTestStatisticData(summer68+1e-6);  double cls68L = res->CLs(), clsb68L = res->CLsplusb();
    res->SetTestStatisticData(winter68+1e-6);  double cls68H = res->CLs(), clsb68H = res->CLsplusb();
    res->SetTestStatisticData(summer95+1e-6);  double cls95L = res->CLs(), clsb95L = res->CLsplusb();
    res->SetTestStatisticData(winter95+1e-6);  double cls95H = res->CLs(), clsb95H = res->CLsplusb();
    /*
    std::cout << "Expected CLs: median " << clsMid 
                                << ", 68% band [" << cls68L << ", " << cls68H << "]" 
                                << ", 95% band [" << cls95L << ", " << cls95H << "]" << std::endl;    
    std::cout << "Expected CLsb: median " << clsbMid 
                                << ", 68% band [" << clsb68L << ", " << clsb68H << "]" 
                                << ", 95% band [" << clsb95L << ", " << clsb95H << "]" << std::endl;    
    */
    b.obs->Set(b.n+1);     b.obs->SetPoint(b.n, x, clsObs);   b.obs->SetPointError(b.n, 0, 0, clsObsErr, clsObsErr);
    b.exp68->Set(b.n+1); b.exp68->SetPoint(b.n, x, clsMid); b.exp68->SetPointError(b.n, 0, 0, clsMid-cls68L, cls68H-clsMid);
    b.exp95->Set(b.n+1); b.exp95->SetPoint(b.n, x, clsMid); b.exp95->SetPointError(b.n, 0, 0, clsMid-cls95L, cls95H-clsMid);
    b.n++;
}

void lepLikePlot() {
}
