// Date   : 30/05/2005
// Author : N.Almeida (LIP)
// falta fazer o update dos block sizes

#ifndef DCCTBEVENTBLOCK_HH
#define DCCTBEVENTBLOCK_HH


#include "DCCBlockPrototype.h"

class DCCTBTowerBlock;
class DCCTBDataParser;
class DCCTBTrailerBlock;
class DCCTBTCCBlock;
class DCCTBSRPBlock;

class DCCTBEventBlock : public DCCTBBlockPrototype {
	
	public :
		
		DCCTBEventBlock(
			DCCTBDataParser * parser, 
			uint32_t * buffer, 
			uint32_t numbBytes, 
			uint32_t wordsToEnd, 
			uint32_t wordBufferOffset = 0 , 
			uint32_t wordEventOffset = 0 
		);
		
		~DCCTBEventBlock();
		
		void dataCheck(); 
		
		std::vector< DCCTBTowerBlock * > & towerBlocks();
		std::vector< DCCTBTCCBlock *   > & tccBlocks();
		DCCTBSRPBlock               * srpBlock();
		DCCTBTrailerBlock           * trailerBlock();
		std::vector< DCCTBTowerBlock * >   towerBlocksById(uint32_t towerId);
		using DCCTBBlockPrototype::compare;
		std::pair<bool,std::string> compare(DCCTBEventBlock * );

		bool eventHasErrors();
		std::string eventErrorString();
		void displayEvent(std::ostream & os=std::cout);
	
		
	protected :
		enum dccFields{ 
			
			PHYSICTRIGGER        = 1,
			CALIBRATIONTRIGGER   = 2,
			TESTTRIGGER          = 3,
			TECHNICALTRIGGER     = 4,
			
			CH_ENABLED           = 0,
			CH_DISABLED          = 1,
			CH_TIMEOUT           = 2,
			CH_SUPPRESS          = 7,
			
			SR_NREAD              = 0,
			
		
			BOE                  = 0x5, 
			
			DCCERROR_EMPTYEVENT  = 0x1, 
			
			TOWERHEADER_SIZE     = 8, 
			TRAILER_SIZE         = 8
	
		
		};		

		std::vector< DCCTBTowerBlock * > towerBlocks_      ;
		std::vector< DCCTBTCCBlock   * > tccBlocks_        ;
		DCCTBTrailerBlock       *   dccTrailerBlock_  ;
		DCCTBSRPBlock           *   srpBlock_;
		uint32_t wordBufferOffset_;
		bool emptyEvent;
};


inline std::vector< DCCTBTowerBlock * > & DCCTBEventBlock::towerBlocks()  { return towerBlocks_;     }
inline std::vector< DCCTBTCCBlock * >   & DCCTBEventBlock::tccBlocks()    { return tccBlocks_;       }
inline DCCTBSRPBlock               * DCCTBEventBlock::srpBlock()     { return srpBlock_;        }
inline DCCTBTrailerBlock           * DCCTBEventBlock::trailerBlock() { return dccTrailerBlock_; }

#endif
