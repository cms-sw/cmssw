#include <CondFormats/JetMETObjects/interface/JetResolutionObject.h>

#include <TH2.h>
#include <TFile.h>

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cout << "Usage: draw_jer_factors <input_file> <output_file>" << std::endl;
        return 1;
    }

    JME::JetResolutionObject factors(argv[1]);
    factors.dump();

    const std::vector<JME::JetResolutionObject::Record> records = factors.getRecords();

    std::vector<float> etas;
    for (const auto& record: records) {
        if (etas.empty()) {
            etas.push_back(record.getBinsRange()[0].min);
            etas.push_back(record.getBinsRange()[0].max);
        } else {
            etas.push_back(record.getBinsRange()[0].max);
        }
    }

    std::vector<float> pts = {8, 10, 12, 15, 18, 21, 24, 28, 32, 37, 43, 49, 56, 64, 74, 84,
     97, 114, 133, 153, 174, 196, 220, 245, 272, 300, 362, 430,
     507, 592, 686, 790, 905, 1032, 1172, 1327, 1497, 1684, 1890, //1999};
     2000, 2238, 2500, 2787, 3103, 3450,5000,7000,10000};

    TH2* plot = new TH2F("plot", "plot", pts.size() - 1, &pts.at(0), etas.size() - 1, &etas.at(0));

    for (size_t i = 0; i < etas.size() - 1; i++) {
        float mean_eta = (etas[i] + etas[i + 1]) / 2;


        for (size_t j = 0; j < pts.size() - 1; j++) {
            float mean_pt = (pts[j] + pts[j + 1]) / 2;

            if (mean_pt * cosh(mean_eta) > 7000)
                continue;

            const JME::JetResolutionObject::Record* record = factors.getRecord(JME::JetParameters().setJetEta(mean_eta));
            if (!record)
                continue;

            plot->SetBinContent(plot->FindBin(mean_pt, mean_eta), factors.evaluateFormula(*record, JME::JetParameters().setJetPt(mean_pt)));
        }

    }

    TFile* output = TFile::Open(argv[2], "recreate");
    plot->Write();
    delete output;

    return 0;
}
