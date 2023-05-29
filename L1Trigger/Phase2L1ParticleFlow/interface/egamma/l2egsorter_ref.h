#ifndef L2EgSorter_REF_H
#define L2EgSorter_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "DataFormats/L1TParticleFlow/interface/pf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/bitonic_hybrid_sort_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <algorithm>

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class L2EgSorterEmulator {
  public:
    L2EgSorterEmulator(unsigned int nBoards, unsigned int nEGPerBoard, unsigned int nEGOut, bool debug)
        : nBOARDS(nBoards), nEGPerBoard(nEGPerBoard), nEGOut(nEGOut), debug_(debug) {}

    L2EgSorterEmulator(const edm::ParameterSet &iConfig);

    virtual ~L2EgSorterEmulator() {}

    template <int NBoards, int NObjs>
    void toFirmware(const std::vector<l1ct::OutputBoard> &in,
                    EGIsoObj (&photons_in)[NBoards][NObjs],
                    EGIsoEleObj (&eles_in)[NBoards][NObjs]) const {
      for (unsigned int ib = 0; ib < NBoards; ib++) {
        const auto &board = in[ib];
        for (unsigned int io = 0; io < NObjs; io++) {
          EGIsoObj pho;
          EGIsoEleObj ele;
          if (io < board.egphoton.size())
            pho = board.egphoton[io];
          else
            pho.clear();
          if (io < board.egelectron.size())
            ele = board.egelectron[io];
          else
            ele.clear();

          photons_in[ib][io] = pho;
          eles_in[ib][io] = ele;
        }
      }
    }

    void toFirmware(const std::vector<EGIsoObjEmu> &out_photons,
                    const std::vector<EGIsoEleObjEmu> &out_eles,
                    EGIsoObj out_egphs[/*nObjOut*/],
                    EGIsoEleObj out_egeles[/*nObjOut*/]) const;

    void run(const std::vector<l1ct::OutputBoard> &in,
             std::vector<EGIsoObjEmu> &out_photons,
             std::vector<EGIsoEleObjEmu> &out_eles) const;

    void setDebug(int verbose) { debug_ = verbose; }

    unsigned int nInputBoards() const { return nBOARDS; }
    unsigned int nInputObjPerBoard() const { return nEGPerBoard; }
    unsigned int nOutputObj() const { return nEGOut; }

  private:
    template <typename T>
    void resize_input(std::vector<T> &in) const {
      if (in.size() > nEGPerBoard) {
        in.resize(nEGPerBoard);
      } else if (in.size() < nEGPerBoard) {
        for (unsigned int i = 0, diff = (nEGPerBoard - in.size()); i < diff; ++i) {
          in.push_back(T());
          in.back().clear();
        }
      }
    }

    template <typename T>
    static bool comparePt(const T &obj1, const T &obj2) {
      return (obj1.hwPt > obj2.hwPt);
    }

    template <typename T>
    void print_objects(const std::vector<T> &objs, const std::string &label) const {
      for (unsigned int i = 0; i < objs.size(); ++i) {
        dbgCout() << label << " [" << i << "] pt: " << objs[i].hwPt << " eta: " << objs[i].hwEta
                  << " phi: " << objs[i].hwPhi << " qual: " << objs[i].hwQual.to_string(2) << std::endl;
      }
    }

    template <typename T>
    void merge_boards(const std::vector<T> &in_board1,
                      const std::vector<T> &in_board2,
                      std::vector<T> &out,
                      unsigned int nOut) const {
      // we crate a bitonic list
      out = in_board1;
      std::reverse(out.begin(), out.end());
      std::copy(in_board2.begin(), in_board2.end(), std::back_inserter(out));
      hybridBitonicMergeRef(&out[0], out.size(), 0, false);

      // std::merge(in_board1.begin(), in_board1.end(), in_board2_copy.begin(), in_board2_copy.end(), std::back_inserter(out), comparePt<T>);
      if (out.size() > nOut)
        out.resize(nOut);
    }

    template <typename T>
    void merge(const std::vector<std::vector<T>> &in_objs, std::vector<T> &out) const {
      if (in_objs.size() == 1) {
        std::copy(in_objs[0].begin(), in_objs[0].end(), std::back_inserter(out));
        if (out.size() > nEGOut)
          out.resize(nEGOut);
      } else if (in_objs.size() == 2) {
        merge_boards(in_objs[0], in_objs[1], out, nEGOut);
      } else {
        std::vector<std::vector<T>> to_merge;
        for (unsigned int id = 0, idn = 1; id < in_objs.size(); id += 2, idn = id + 1) {
          if (idn >= in_objs.size()) {
            to_merge.push_back(in_objs[id]);
          } else {
            std::vector<T> pair_merge;
            merge_boards(in_objs[id], in_objs[idn], pair_merge, nEGPerBoard);
            to_merge.push_back(pair_merge);
          }
        }
        merge(to_merge, out);
      }
    }

    const unsigned int nBOARDS;
    const unsigned int nEGPerBoard;
    const unsigned int nEGOut;
    int debug_;
  };
}  // namespace l1ct

#endif
