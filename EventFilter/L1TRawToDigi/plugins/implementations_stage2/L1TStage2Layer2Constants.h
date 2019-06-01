#ifndef L1TCaloStage2Layer2Constants_h
#define L1TCaloStage2Layer2Constants_h

namespace l1t {

  namespace stage2 {

    namespace layer2 {

      extern const signed int fedId;

      namespace mp {

        extern const unsigned int offsetBoardId;

        extern const unsigned int nInputFramePerBX;
        extern const unsigned int nOutputFramePerBX;

      }  // namespace mp

      namespace demux {

        extern const unsigned int nOutputFramePerBX;

        extern const unsigned int nEGPerLink;
        extern const unsigned int nTauPerLink;
        extern const unsigned int nJetPerLink;
        extern const unsigned int nEtSumPerLink;

        extern const unsigned int amcSlotNum;

      }  // namespace demux

    }  // namespace layer2

  }  // namespace stage2

}  // namespace l1t

#endif
