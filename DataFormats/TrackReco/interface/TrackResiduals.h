#ifndef TrackReco_TrackResiduals_h
#define TrackReco_TrackResiduals_h

#include <vector>

namespace reco {

  class TrackResiduals {
  public:
    // In principle 8bits would suffice for histos...
    using StorageType = unsigned short;

    TrackResiduals() {}
    void resize(unsigned int nHits) { m_storage.resize(4 * nHits); }
    void setResidualXY(int idx, float residualX, float residualY);
    void setPullXY(int idx, float pullX, float pullY);
    /// get the residual of the ith hit
    float residualX(int i) const { return unpack_residual(m_storage[4 * i]); }
    float residualY(int i) const { return unpack_residual(m_storage[4 * i + 1]); }
    float pullX(int i) const { return unpack_pull(m_storage[4 * i + 2]); }
    float pullY(int i) const { return unpack_pull(m_storage[4 * i + 3]); }

  private:
    static float unpack_pull(StorageType);
    static StorageType pack_pull(float);
    static float unpack_residual(StorageType);
    static StorageType pack_residual(float);

  private:
    std::vector<StorageType> m_storage;
  };

}  // namespace reco

#endif
