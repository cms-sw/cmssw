#include <CondFormats/JetMETObjects/interface/JetResolutionObject.h>

int main(int argc, char **argv) {

    JME::JetResolution jer(argv[1]);

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

    std::vector<float> pts = {1, 10, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 2000, 10000};

    for (size_t i = 0; i < etas.size() - 1; i++) {
        float mean_eta = (etas[i] + etas[i + 1]) / 2;
        for (float pt: pts) {
            std::cout << "eta: " << mean_eta << "  pt: " << pt << " -> jer = " << jer.getResolution(JME::JetParameters().setJetPt(pt).setJetEta(mean_eta)) << std::endl;
        }
    }

    return 0;
}
