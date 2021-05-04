// Class for Resistive Plate Chamber (RPC) Data Record

#ifndef __l1t_emtf_RPC_h__
#define __l1t_emtf_RPC_h__

#include <vector>
#include <cstdint>

namespace l1t {
  namespace emtf {
    class RPC {
    public:
      explicit RPC(uint64_t dataword);

      RPC()
          : phi(-99),
            theta(-99),
            word(-99),
            frame(-99),
            link(-99),
            rpc_bxn(-99),
            bc0(-99),
            tbin(-99),
            vp(-99),
            format_errors(0),
            dataword(-99){};

      virtual ~RPC(){};

      void set_phi(int bits) { phi = bits; }
      void set_theta(int bits) { theta = bits; }
      void set_word(int bits) { word = bits; }
      void set_frame(int bits) { frame = bits; }
      void set_link(int bits) { link = bits; }
      void set_rpc_bxn(int bits) { rpc_bxn = bits; }
      void set_bc0(int bits) { bc0 = bits; }
      void set_tbin(int bits) { tbin = bits; }
      void set_vp(int bits) { vp = bits; }
      void add_format_error() { format_errors += 1; }
      void set_dataword(uint64_t bits) { dataword = bits; }

      int Phi() const { return phi; }
      int Theta() const { return theta; }
      int Word() const { return word; }
      int Frame() const { return frame; }
      int Link() const { return link; }
      int RPC_BXN() const { return rpc_bxn; }
      int BC0() const { return bc0; }
      int TBIN() const { return tbin; }
      int VP() const { return vp; }
      int Format_errors() const { return format_errors; }
      uint64_t Dataword() const { return dataword; }

    private:
      int phi;
      int theta;
      int word;
      int frame;
      int link;
      int rpc_bxn;
      int bc0;
      int tbin;
      int vp;
      int format_errors;
      uint64_t dataword;

    };  // End of class RPC

    // Define a vector of RPC
    typedef std::vector<RPC> RPCCollection;

  }  // End of namespace emtf
}  // End of namespace l1t

#endif /* define __l1t_emtf_RPC_h__ */
