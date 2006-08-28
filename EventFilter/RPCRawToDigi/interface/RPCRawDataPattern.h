#ifndef RPCRawDataPattern_h
#define RPCRawDataPattern_h

namespace rpcraw{
	namespace bx{
		static const int BX_MASK  = 0XC;
		static const int BX_SHIFT = 0;
	}
	
	namespace tb_link{
		static const int TB_LINK_INPUT_NUMBER_MASK  = 0X1F;
		static const int TB_LINK_INPUT_NUMBER_SHIFT =0;

		static const int TB_RMB_MASK = 0X3F;
		static const int TB_RMB_SHIFT =5;
	}
	namespace lb{
		static const int PARTITION_DATA_MASK  = 0XFF;
		static const int PARTITION_DATA_SHIFT =0;

		static const int HALFP_MASK = 0X1;
		static const int HALFP_SHIFT =8;

		static const int EOD_MASK = 0X1;
		static const int EOD_SHIFT =9;

		static const int PARTITION_NUMBER_MASK = 0XF;
		static const int PARTITION_NUMBER_SHIFT =10;

		static const int LB_MASK = 0X3;
		static const int LB_SHIFT =14;
	}
	namespace bits{
		static const int BITS_PER_PARTITION=8;
	}
	namespace error{
		static const int TB_LINK_MASK  = 0X1F;
		static const int TB_LINK_SHIFT =0;

		static const int TB_RMB_MASK = 0X3F;
		static const int TB_RMB_SHIFT =5;
	}
}

#endif
