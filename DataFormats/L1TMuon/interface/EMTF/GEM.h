// Class for Gas-Electron Multiplier (GEM) EMTF Data Record

#ifndef __l1t_emtf_GEM_h__
#define __l1t_emtf_GEM_h__

#include <vector>
#include <cstdint>

namespace l1t {
  namespace emtf {
    class GEM {
    public:
      explicit GEM(const uint64_t dataword);

      GEM()
          : pad(-99),
            partition(-99),
            cluster_size(-99),
            cluster_id(-99),
            link(-99),
            gem_bxn(-99),
            bc0(-99),
            tbin(-99),
            vp(-99),
            format_errors(0),
            dataword(-99){};

      virtual ~GEM() = default;

      inline void set_pad(const int bits) { pad = bits; }
      inline void set_partition(const int bits) { partition = bits; }
      inline void set_cluster_size(const int bits) { cluster_size = bits; }
      inline void set_cluster_id(const int bits) { cluster_id = bits; }
      inline void set_link(const int bits) { link = bits; }
      inline void set_gem_bxn(const int bits) { gem_bxn = bits; }
      inline void set_bc0(const int bits) { bc0 = bits; }
      inline void set_tbin(const int bits) { tbin = bits; }
      inline void set_vp(const int bits) { vp = bits; }
      inline void add_format_error() { format_errors += 1; }
      inline void set_dataword(const uint64_t bits) { dataword = bits; }

      /// Returns the lowest pad (strip pair, i.e., local phi) of the cluster
      inline int Pad() const { return pad; }
      /// Returns the eta partition (local eta) of the cluster
      inline int Partition() const { return partition; }
      /// Returns the size (in pads) of the cluster
      inline int ClusterSize() const { return cluster_size; }
      /// Returns the the cluster ID within the link
      inline int ClusterID() const { return cluster_id; }
      /// Returns the input link of the cluster
      inline int Link() const { return link; }
      /// Returns the BX ID of the cluster
      inline int GEM_BXN() const { return gem_bxn; }
      /// Returns whether the cluster has BC0
      inline int BC0() const { return bc0; }
      /// Returns the time bin of the cluster
      inline int TBIN() const { return tbin; }
      /// Returns the valid flag? of the cluster
      inline int VP() const { return vp; }
      /// Returns the format errors associated with the cluster
      inline int Format_errors() const { return format_errors; }
      /// Returns the raw data word of the cluster
      inline uint64_t Dataword() const { return dataword; }

    private:
      int pad;            ///< Pad (strip pair, i.e., local phi) of the GEM cluster
      int partition;      ///< Partition (local eta) of the GEM cluster
      int cluster_size;   ///< Size (in pads) of the GEM cluster
      int cluster_id;     ///< Cluster number of the GEM cluster
      int link;           ///< Input GEM link of the GEM cluster
      int gem_bxn;        ///< BX ID of the GEM cluster
      int bc0;            ///< BC0 valid for the GEM cluster
      int tbin;           ///< Time bin of the GEM cluster
      int vp;             ///< Valid status? of the GEM cluster
      int format_errors;  ///< Number of format errors for the GEM cluster
      uint64_t dataword;  ///< Raw EMTF DAQ word for the GEM cluster

    };  // End of class GEM

    // Define a vector of GEM
    typedef std::vector<GEM> GEMCollection;

  }  // End of namespace emtf
}  // End of namespace l1t

#endif /* define __l1t_emtf_GEM_h__ */
