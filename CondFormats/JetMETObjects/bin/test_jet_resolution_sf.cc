#include <CondFormats/JetMETObjects/interface/JetResolutionObject.h>

int main(int argc, char **argv) {

    JME::JetResolutionScaleFactor jer(argv[1]);
    jer.dump();

    const std::vector<JME::JetResolutionObject::Record> records = jer.getResolutionObject()->getRecords();

    std::vector<float> etas;
    for (const auto& record: records) {
        if (etas.empty()) {
            etas.push_back(record.getBinsRange()[0].min);
            etas.push_back(record.getBinsRange()[0].max);
        } else {
            etas.push_back(record.getBinsRange()[0].max);
        }
    }

    for (size_t i = 0; i < etas.size() - 1; i++) {
        float mean_eta = (etas[i] + etas[i + 1]) / 2;
        JME::JetParameters params;
        params.setJetEta(mean_eta);
        std::cout << "eta: " << mean_eta << " -> SF / UP / DOWN = " << jer.getScaleFactor(params) << " / " << jer.getScaleFactor(params, Variation::UP) << " / " << jer.getScaleFactor(params, Variation::DOWN) << std::endl;
    }

    return 0;
}
