#include "CondFormats/Serialization/interface/SerializationTest.h"

#include "CondFormats/SiStripObjects/interface/Serialization.h"

int main()
{
    testSerialization<FedChannelConnection>();
    testSerialization<SiStripApvGain>();
    testSerialization<SiStripBackPlaneCorrection>();
    testSerialization<SiStripBadStrip>();
    testSerialization<SiStripBadStrip::DetRegistry>();
    testSerialization<SiStripBaseDelay>();
    testSerialization<SiStripBaseDelay::Delay>();
    testSerialization<SiStripConfObject>();
    testSerialization<SiStripDetVOff>();
    testSerialization<SiStripFedCabling>();
    testSerialization<SiStripFedCabling::Conns>();
    testSerialization<SiStripLatency>();
    testSerialization<SiStripLatency::Latency>();
    testSerialization<SiStripLorentzAngle>();
    testSerialization<SiStripNoises>();
    testSerialization<SiStripNoises::DetRegistry>();
    testSerialization<SiStripPedestals>();
    testSerialization<SiStripPedestals::DetRegistry>();
    testSerialization<SiStripRunSummary>();
    testSerialization<SiStripSummary>();
    testSerialization<SiStripSummary::DetRegistry>();
    testSerialization<SiStripThreshold>();
    testSerialization<SiStripThreshold::Container>();
    testSerialization<SiStripThreshold::Data>();
    testSerialization<SiStripThreshold::DetRegistry>();
    testSerialization<std::vector<std::vector<FedChannelConnection>>>();
    testSerialization<std::vector<FedChannelConnection>>();
    testSerialization<std::vector<SiStripBadStrip::DetRegistry>>();
    testSerialization<std::vector<SiStripBaseDelay::Delay>>();
    testSerialization<std::vector<SiStripLatency::Latency>>();
    testSerialization<std::vector<SiStripNoises::DetRegistry>>();
    testSerialization<std::vector<SiStripPedestals::DetRegistry>>();
    testSerialization<std::vector<SiStripSummary::DetRegistry>>();
    testSerialization<std::vector<SiStripThreshold::Container>>();
    testSerialization<std::vector<SiStripThreshold::Data>>();
    testSerialization<std::vector<SiStripThreshold::DetRegistry>>();

    return 0;
}
