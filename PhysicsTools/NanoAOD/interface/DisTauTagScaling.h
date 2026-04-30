namespace scaling {
  struct PfCand {
    inline static const std::vector<std::vector<float>> mean = {{0, 0},
                                                                {50.0, 50.0},
                                                                {0.0, 0.0},
                                                                {0.0, 0.0},
                                                                {0.09469, 0.09469},
                                                                {0, 0},
                                                                {0, 0},
                                                                {0, 0},
                                                                {5.0, 5.0},
                                                                {5.0, 5.0},
                                                                {6.442, 6.442},
                                                                {0, 0},
                                                                {-0.01635, -0.01635},
                                                                {0.02704, 0.02704},
                                                                {-0.01443, -0.01443},
                                                                {0.05551, 0.05551},
                                                                {11.16, 11.16},
                                                                {13.53, 13.53},
                                                                {0.5, 0.5},
                                                                {0.5, 0.5},
                                                                {1.3, 1.3},
                                                                {0.5, 0.5},
                                                                {0, 0},
                                                                {0, 0}};
    inline static const std::vector<std::vector<float>> std = {{1, 1},           {50.0, 50.0},
                                                               {3.0, 3.0},       {3.141592653589793, 3.141592653589793},
                                                               {0.0651, 0.0651}, {1, 1},
                                                               {1, 1},           {1, 1},
                                                               {5.0, 5.0},       {5.0, 5.0},
                                                               {8.344, 8.344},   {1, 1},
                                                               {2.4, 2.4},       {0.08913, 0.08913},
                                                               {7.444, 7.444},   {0.5998, 0.5998},
                                                               {60.39, 60.39},   {6.44, 6.44},
                                                               {0.5, 0.5},       {0.5, 0.5},
                                                               {1.3, 1.3},       {0.5, 0.5},
                                                               {1, 1},           {1, 1}};
    inline static const std::vector<std::vector<float>> lim_min = {
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-5, -5},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-5, -5},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-5, -5},
        {-5, -5},
        {-5, -5},
        {-5, -5},
        {-5, -5},
        {-5, -5},
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-1.0, -1.0},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}};
    inline static const std::vector<std::vector<float>> lim_max = {
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {1.0, 1.0},
        {1.0, 1.0},
        {1.0, 1.0},
        {5, 5},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {1.0, 1.0},
        {1.0, 1.0},
        {5, 5},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {5, 5},
        {5, 5},
        {5, 5},
        {5, 5},
        {5, 5},
        {5, 5},
        {1.0, 1.0},
        {1.0, 1.0},
        {1.0, 1.0},
        {1.0, 1.0},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}};
  };
  struct PfCandCategorical {
    inline static const std::vector<std::vector<float>> mean = {{0, 0}, {0, 0}, {0, 0}};
    inline static const std::vector<std::vector<float>> std = {{1, 1}, {1, 1}, {1, 1}};
    inline static const std::vector<std::vector<float>> lim_min = {
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}};
    inline static const std::vector<std::vector<float>> lim_max = {
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}};
  };
};  // namespace scaling

namespace setupSizes {
  const inline long int output_classes = 2;
  const inline size_t n_Global = 5;
  const inline size_t n_PfCand = 24;
  const inline size_t nSeq_PfCand = 50;
  const inline size_t n_PfCandCategorical = 3;
  const inline size_t nSeq_PfCandCategorical = 50;
  const inline std::vector<std::string> CellObjectTypes{"PfCand", "PfCandCategorical"};
};  // namespace setupSizes

enum class Global_Features { jet_pt = 0, jet_eta = 1, Lxy = 2, Lz = 3, Lrel = 4 };

enum class PfCand_Features {
  pfCand_valid = 0,
  pfCand_pt = 1,
  pfCand_eta = 2,
  pfCand_phi = 3,
  pfCand_mass = 4,
  pfCand_charge = 5,
  pfCand_puppiWeight = 6,
  pfCand_puppiWeightNoLep = 7,
  pfCand_lostInnerHits = 8,
  pfCand_nPixelHits = 9,
  pfCand_nHits = 10,
  pfCand_hasTrackDetails = 11,
  pfCand_dxy = 12,
  pfCand_dxy_error = 13,
  pfCand_dz = 14,
  pfCand_dz_error = 15,
  pfCand_track_chi2 = 16,
  pfCand_track_ndof = 17,
  pfCand_caloFraction = 18,
  pfCand_hcalFraction = 19,
  pfCand_rawCaloFraction = 20,
  pfCand_rawHcalFraction = 21,
  pfCand_deta = 22,
  pfCand_dphi = 23
};

enum class PfCandCategorical_Features { pfCand_particleType = 0, pfCand_pvAssociationQuality = 1, pfCand_fromPV = 2 };

enum class CellObjectType { PfCand, PfCandCategorical };

template <typename T>
struct FeaturesHelper;

template <>
struct FeaturesHelper<PfCand_Features> {
  static constexpr CellObjectType object_type = CellObjectType::PfCand;
  static constexpr size_t size = 24;
  static constexpr size_t length = 50;
  using scaler_type = scaling::PfCand;
};

template <>
struct FeaturesHelper<PfCandCategorical_Features> {
  static constexpr CellObjectType object_type = CellObjectType::PfCandCategorical;
  static constexpr size_t size = 3;
  static constexpr size_t length = 50;
  using scaler_type = scaling::PfCandCategorical;
};

using FeatureTuple = std::tuple<PfCand_Features, PfCandCategorical_Features>;

enum class PFParticleType {
  Undefined = 0,  // undefined
  h = 1,          // charged hadron
  e = 2,          // electron
  mu = 3,         // muon
  gamma = 4,      // photon
  h0 = 5,         // neutral hadron
  h_HF = 6,       // HF tower identified as a hadron
  egamma_HF = 7,  // HF tower identified as an EM particle
};

inline PFParticleType TranslatePdgIdToPFParticleType(int pdgId) {
  static const std::map<int, PFParticleType> type_map = {
      {11, PFParticleType::e},
      {13, PFParticleType::mu},
      {22, PFParticleType::gamma},
      {211, PFParticleType::h},
      {130, PFParticleType::h0},
      {1, PFParticleType::h_HF},
      {2, PFParticleType::egamma_HF},
  };
  auto iter = type_map.find(std::abs(pdgId));
  return iter == type_map.end() ? PFParticleType::Undefined : iter->second;
}
