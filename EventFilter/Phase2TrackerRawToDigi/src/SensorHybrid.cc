#include "EventFilter/Phase2TrackerRawToDigi/interface/SensorHybrid.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"  // Add this line

SensorHybrid::SensorHybrid(const DetId& det_id,
                           edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_1,
                           edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator sensor_2,
                           const edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator nullIter,
                           const bool cic_id,
                           const TrackerGeometry& trackerGeometry,
                           const unsigned int eventId)
    : cic_id_(cic_id), eventId_(eventId) {
  const unsigned int cic_boundary_1 = set_sensor_type(det_id+1, trackerGeometry, 1);
  const unsigned int cic_boundary_2 = set_sensor_type(det_id+2, trackerGeometry, 2);
  // Sensors containing no clusters are missing from the DetSetVector, so must protect against this.
  if (sensor_1 != nullIter)
    sensor_1_clusters_ = get_clusters_on_cic(sensor_1, cic_boundary_1);
  if (sensor_2 != nullIter)
    sensor_2_clusters_ = get_clusters_on_cic(sensor_2, cic_boundary_2);
}

unsigned int SensorHybrid::set_sensor_type(
    const DetId& det_id,
    const TrackerGeometry& trackerGeometry,
    const int internal_id) {
  using namespace Phase2TrackerSpecifications;

  const GeomDetUnit* sensor_unit = trackerGeometry.idToDetUnit(det_id);
  unsigned int cic_boundary_in_z = CIC_Z_BOUNDARY_STRIPS;

  if (sensor_unit == nullptr)
    throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << "DetUnit not found";

  TrackerGeometry::ModuleType moduleType = trackerGeometry.getDetectorType(det_id);
  switch (moduleType) {
    case TrackerGeometry::ModuleType::Ph2PSS:
      if (internal_id == 1) {
        sensor_type_1 = TrackerGeometry::ModuleType::Ph2PSS;
      } else if (internal_id == 2) {
        sensor_type_2 = TrackerGeometry::ModuleType::Ph2PSS;
      }
      break;

    case TrackerGeometry::ModuleType::Ph2SS:
      if (internal_id == 1) {
        sensor_type_1 = TrackerGeometry::ModuleType::Ph2SS;
      } else if (internal_id == 2) {
        sensor_type_2 = TrackerGeometry::ModuleType::Ph2SS;
      }
      break;

    case TrackerGeometry::ModuleType::Ph2PSP:
      if (internal_id == 1) {
        sensor_type_1 = TrackerGeometry::ModuleType::Ph2PSP;
      } else if (internal_id == 2) {
        sensor_type_2 = TrackerGeometry::ModuleType::Ph2PSP;
      }
      cic_boundary_in_z = CIC_Z_BOUNDARY_PIXEL;
      break;

    default:
      throw cms::Exception("InvalidModuleType")
          << "Unexpected TrackerGeometry::ModuleType for detId: " << det_id << ".";
  }
  return cic_boundary_in_z;  
}

std::vector<Phase2TrackerCluster1D*> SensorHybrid::get_clusters_on_cic(
    edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator clusterIterator,
    const unsigned int cic_boundary_in_z
    ) {
  using namespace Phase2TrackerSpecifications;

  std::vector<Phase2TrackerCluster1D*> filteredClusters;

  for (auto& cluster : *clusterIterator) {
    if (cic_id_ == true && cluster.column() > cic_boundary_in_z && cluster.size() <=7 ) {
      filteredClusters.push_back(&cluster);
    } else if (cic_id_ == false && cluster.column() <= cic_boundary_in_z  && cluster.size() <=7 ) {
      filteredClusters.push_back(&cluster);
    }
  }

  return filteredClusters;
}

unsigned int SensorHybrid::get_number_of_strip_clusters() {
  if (sensor_type_1 == TrackerGeometry::ModuleType::Ph2PSP || sensor_type_2 == TrackerGeometry::ModuleType::Ph2PSS) {
    return sensor_2_clusters_.size();
  } else if (sensor_type_1 == TrackerGeometry::ModuleType::Ph2SS) {
    return sensor_1_clusters_.size() + sensor_2_clusters_.size();
  }
  return 0;
}

unsigned int SensorHybrid::get_number_of_pixel_clusters() {
  if (sensor_type_1 == TrackerGeometry::ModuleType::Ph2PSP || sensor_type_2 == TrackerGeometry::ModuleType::Ph2PSS) {
    return sensor_1_clusters_.size();
  } else if (sensor_type_1 == TrackerGeometry::ModuleType::Ph2SS) {
    return 0;
  }
  return 0;
}

unsigned int SensorHybrid::get_payload_size() {
  using namespace Phase2DAQFormatSpecification;
  auto nlines_float =
      std::div(get_number_of_strip_clusters() * SS_CLUSTER_BITS + get_number_of_pixel_clusters() * PX_CLUSTER_BITS,
               N_BITS_PER_WORD);
  return nlines_float.rem > 0 ? nlines_float.quot + 1 + 1 : nlines_float.quot + 1;
}

void SensorHybrid::get_channel_cluster_payload(std::vector<Phase2DAQFormatSpecification::Word32Bits>& payload) {
  using namespace Phase2DAQFormatSpecification;
  using namespace Phase2TrackerSpecifications;

  uint32_t currentWord = 0;
  int bitsFilled = 0;

  if (sensor_type_1 == TrackerGeometry::ModuleType::Ph2PSP || sensor_type_2 == TrackerGeometry::ModuleType::Ph2PSS) {
    // For PS, sensor_2 is always strip and sensor_1 is always pixel

    for (auto& cluster : sensor_2_clusters_) {
      // cluster info
      uint32_t chipID = std::div(cluster->firstStrip(), STRIPS_PER_SSA).quot & CHIP_ID_MAX_VALUE;  // 3 bits
      uint32_t sclusterAddress =
          std::div(cluster->firstStrip(), STRIPS_PER_SSA).rem & SCLUSTER_ADDRESS_PS_MAX_VALUE;  // 7 bits
      uint32_t width = cluster->size() & WIDTH_MAX_VALUE;                                       // 3 bits
      uint32_t mipBit = cluster->threshold() & 0x1;                                             // 1 bit

      uint32_t clusterData = (chipID << (SS_CLUSTER_BITS - CHIP_ID_BITS)) |
                             (sclusterAddress << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) |
                             (width << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) |
                             mipBit;

      if (bitsFilled + SS_CLUSTER_BITS <= N_BITS_PER_WORD) {
        currentWord |= clusterData << (N_BITS_PER_WORD - bitsFilled - SS_CLUSTER_BITS);
        bitsFilled += SS_CLUSTER_BITS;
      } else {
        int bitsLeft = N_BITS_PER_WORD - bitsFilled;
        currentWord |= clusterData >> (SS_CLUSTER_BITS - bitsLeft);
        payload.push_back(currentWord);

        currentWord = clusterData << (N_BITS_PER_WORD - (SS_CLUSTER_BITS - bitsLeft));
        bitsFilled = SS_CLUSTER_BITS - bitsLeft;
      }
    }

    for (auto& cluster : sensor_1_clusters_) {
      // cluster info
      uint32_t chipID = std::div(cluster->firstStrip(), STRIPS_PER_SSA).quot & CHIP_ID_MAX_VALUE;  // 3 bits
      uint32_t sclusterAddress =
          std::div(cluster->firstStrip(), STRIPS_PER_SSA).rem & SCLUSTER_ADDRESS_PS_MAX_VALUE;  // 7 bits
      uint32_t width = cluster->size() & WIDTH_MAX_VALUE;                                       // 3 bits
      uint32_t z = cluster->column() & 0xF;  // 4 bits (ensure it's within 4-bit range)

      // Encode cluster data into 17 bits: 3 (chipID) + 7 (sclusterAddress) + 3 (width) + 4 (z)
      uint32_t clusterData = (chipID << (PX_CLUSTER_BITS - CHIP_ID_BITS)) |
                             (sclusterAddress << (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS)) |
                             (width << (PX_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_PS - WIDTH_BITS)) | z;

      if (bitsFilled + PX_CLUSTER_BITS <= N_BITS_PER_WORD) {
        currentWord |= clusterData << (N_BITS_PER_WORD - bitsFilled - PX_CLUSTER_BITS);
        bitsFilled += PX_CLUSTER_BITS;
      } else {
        int bitsLeft = N_BITS_PER_WORD - bitsFilled;
        currentWord |= clusterData >> (PX_CLUSTER_BITS - bitsLeft);
        payload.push_back(currentWord);

        currentWord = clusterData << (N_BITS_PER_WORD - (PX_CLUSTER_BITS - bitsLeft));
        bitsFilled = PX_CLUSTER_BITS - bitsLeft;
      }
    }

    if (bitsFilled > 0) {
      payload.push_back(currentWord);
    }

  } else if (sensor_type_1 == TrackerGeometry::ModuleType::Ph2SS &&
             sensor_type_2 == TrackerGeometry::ModuleType::Ph2SS) {
    // For SS, both sensors are strip

    for (auto& cluster : sensor_2_clusters_) {
      // cluster info
      uint32_t chipID = std::div(cluster->firstStrip(), STRIPS_PER_CBC).quot & CHIP_ID_MAX_VALUE;  // 3 bits
      uint32_t baseValue = std::div(cluster->firstStrip(), STRIPS_PER_CBC).rem & SCLUSTER_ADDRESS_2S_MAX_VALUE;
      uint32_t lsb = 0;
      uint32_t sclusterAddress = (baseValue << 1) | lsb;   // Combine 7 bits and LSB into 8 bits
      uint32_t width = cluster->size() & WIDTH_MAX_VALUE;  // 3 bits

      uint32_t clusterData = (chipID << (SS_CLUSTER_BITS - CHIP_ID_BITS)) |
                             (sclusterAddress << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_2S)) | width;
      if (bitsFilled + SS_CLUSTER_BITS <= N_BITS_PER_WORD) {
        currentWord |= clusterData << (N_BITS_PER_WORD - bitsFilled - SS_CLUSTER_BITS);
        bitsFilled += SS_CLUSTER_BITS;
      } else {
        int bitsLeft = N_BITS_PER_WORD - bitsFilled;
        currentWord |= clusterData >> (SS_CLUSTER_BITS - bitsLeft);
        payload.push_back(currentWord);

        currentWord = clusterData << (N_BITS_PER_WORD - (SS_CLUSTER_BITS - bitsLeft));
        bitsFilled = SS_CLUSTER_BITS - bitsLeft;
      }
    }

    for (auto& cluster : sensor_1_clusters_) {
      // cluster info
      uint32_t chipID = std::div(cluster->firstStrip(), STRIPS_PER_CBC).quot & CHIP_ID_MAX_VALUE;  // 3 bits
      uint32_t baseValue = std::div(cluster->firstStrip(), STRIPS_PER_CBC).rem & SCLUSTER_ADDRESS_2S_MAX_VALUE;
      uint32_t lsb = 1;
      uint32_t sclusterAddress = (baseValue << 1) | lsb;   // Combine 7 bits and LSB into 8 bits
      uint32_t width = cluster->size() & WIDTH_MAX_VALUE;  // 3 bits

      uint32_t clusterData = (chipID << (SS_CLUSTER_BITS - CHIP_ID_BITS)) |
                             (sclusterAddress << (SS_CLUSTER_BITS - CHIP_ID_BITS - SCLUSTER_ADDRESS_BITS_2S)) | width;

      if (bitsFilled + SS_CLUSTER_BITS <= N_BITS_PER_WORD) {
        currentWord |= clusterData << (N_BITS_PER_WORD - bitsFilled - SS_CLUSTER_BITS);
        bitsFilled += SS_CLUSTER_BITS;
      } else {
        int bitsLeft = N_BITS_PER_WORD - bitsFilled;
        currentWord |= clusterData >> (SS_CLUSTER_BITS - bitsLeft);
        payload.push_back(currentWord);

        currentWord = clusterData << (N_BITS_PER_WORD - (SS_CLUSTER_BITS - bitsLeft));
        bitsFilled = SS_CLUSTER_BITS - bitsLeft;
      }
    }

    if (bitsFilled > 0) {
      payload.push_back(currentWord);
    }
  } else {
    edm::LogError("SensorHybrid") << "Sensors 1 and 2 have inconsistent types";
  }
}

void SensorHybrid::get_payload(std::vector<Phase2DAQFormatSpecification::Word32Bits>& payload) {
  using namespace Phase2DAQFormatSpecification;
  using namespace Phase2TrackerSpecifications;

  // Extracting values
  uint32_t eventID = CIC_CONSTANT_EVENT_ID & L1ID_MAX_VALUE;  // 9 bits for eventId_
  uint32_t channelErrors = 0;                    // 9 bits for errors, set to 0
  uint32_t num_strip_clusters =
      get_number_of_strip_clusters() & ((1 << N_STRIP_CLUSTER_BITS) - 1);  // 7 bits for strip clusters
  uint32_t num_pixel_clusters =
      get_number_of_pixel_clusters() & ((1 << N_PIXEL_CLUSTER_BITS) - 1);  // 7 bits for pixel clusters

  // Combine fields into the 32-bit header
  uint32_t header_ = (eventID << (N_BITS_PER_WORD - L1ID_BITS)) |
                     (channelErrors << (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS)) |
                     (num_strip_clusters << (N_BITS_PER_WORD - L1ID_BITS - CIC_ERROR_BITS - N_STRIP_CLUSTER_BITS)) |
                     (num_pixel_clusters);

  // Convert to Word32Bits and add to payload
  Word32Bits header(header_);
  payload.push_back(header);

  get_channel_cluster_payload(payload);
}
