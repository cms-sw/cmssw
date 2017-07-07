#ifndef L1TCaloStage2Layer2Constants_h
#define L1TCaloStage2Layer2Constants_h


namespace l1t {

  namespace stage2 {

    namespace layer2 {

      namespace mp {

	extern unsigned int offsetBoardId;

	extern unsigned int nInputFramePerBX;
	extern unsigned int nOutputFramePerBX;

      }

      namespace demux {
      
	extern unsigned int nOutputFramePerBX;
	
	extern unsigned int nEGPerLink;
	extern unsigned int nTauPerLink;
	extern unsigned int nJetPerLink;
	extern unsigned int nEtSumPerLink;
      
      }

    }
    
  }

}

#endif
