#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

/**
 * Helper that stores one parameter for each layer/ring (wrapper around std::map<std::vector<double>>)
 */
class SiStripFakeAPVParameters {
public:
  using index = std::pair<int,int>;

  SiStripFakeAPVParameters() {}

  /// Fills the parameters read from cfg and matching the name in the map
  SiStripFakeAPVParameters(const edm::ParameterSet& pset, const std::string& parameterName)
  {
    const int layersTIB = 4;
    const int ringsTID = 3;
    const int layersTOB = 6;
    const int ringsTEC = 7;

    fillSubDetParameter(pset.getParameter<std::vector<double>>(parameterName+"TIB"), int(StripSubdetector::TIB), layersTIB );
    fillSubDetParameter(pset.getParameter<std::vector<double>>(parameterName+"TID"), int(StripSubdetector::TID), ringsTID );
    fillSubDetParameter(pset.getParameter<std::vector<double>>(parameterName+"TOB"), int(StripSubdetector::TOB), layersTOB );
    fillSubDetParameter(pset.getParameter<std::vector<double>>(parameterName+"TEC"), int(StripSubdetector::TEC), ringsTEC );
  }

  inline double get(const index& idx) const
  {
    return m_data.at(idx.first)[idx.second];
  }

  static index getIndex(const TrackerTopology* tTopo, DetId id)
  {
    int layerId{0};
    const int subId =  StripSubdetector(id).subdetId();
    switch(subId) {
      case int(StripSubdetector::TIB):
        layerId = tTopo->tibLayer(id) - 1;
        break;
      case int(StripSubdetector::TOB):
        layerId = tTopo->tobLayer(id) - 1;
        break;
      case int(StripSubdetector::TID):
        layerId = tTopo->tidRing(id) - 1;
        break;
      case int(StripSubdetector::TEC):
        layerId = tTopo->tecRing(id) - 1;
        break;
      default:
        break;
    }
    return std::make_pair(subId, layerId);
  }
private:
  using LayerParameters = std::vector<double>;
  using SubdetParameters = std::map<int,LayerParameters>;
  SubdetParameters m_data;

  /**
   * Fills the map with the paramters for the given subdetector. <br>
   * Each vector "v" holds the parameters for the layers/rings, if the vector has only one parameter
   * all the layers/rings get that parameter. <br>
   * The only other possibility is that the number of parameters equals the number of layers, otherwise
   * an exception of type "Configuration" will be thrown.
   */
  void fillSubDetParameter(const std::vector<double>& v, const int subDet, const unsigned short layers)
  {
    if( v.size() == layers ) {
      m_data.insert(std::make_pair( subDet, v ));
    }
    else if( v.size() == 1 ) {
      LayerParameters parV(layers, v[0]);
      m_data.insert(std::make_pair( subDet, parV ));
    }
    else {
      throw cms::Exception("Configuration") << "ERROR: number of parameters for subDet " << subDet << " are " << v.size() << ". They must be either 1 or " << layers << std::endl;
    }
  }
};
