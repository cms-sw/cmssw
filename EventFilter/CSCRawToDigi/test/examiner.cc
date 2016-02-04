#include <stdio.h>
#include <errno.h>
#include "EventFilter/CSCRawToDigi/src/CSCDCCExaminer.cc"
// To compile: g++ -o examiner examiner.cc -I../../../

int main(int argc, char* argv[]){
	CSCDCCExaminer examiner;
	examiner.modeDDU(true);
	examiner.crcCFEB(0);
	examiner.crcTMB (0);
	examiner.crcALCT(0);
	examiner.output1().show();
	examiner.output2().show();

	if(argc!=2){
		printf("Usage: ./examiner INPUT_FILENAME\n");
		return 0;
	}
	FILE *input;
	if( (input=fopen(argv[1],"rt"))==NULL ){
		printf("Cannot open input file: %s (errno=%d)\n",argv[2],errno);
		return 1;
	}

	// Arbitrary size chunk of data to be read from file
	const    int   bufferSize = 116384;
	unsigned short buffer[bufferSize];
	int length = 0;
	bzero(buffer,sizeof(buffer));

	// Read entire data block from file;
	//  don't bother about DDU/DCC headers and traiers, as they are going to found by the examiner
	while( !feof(input) && (length=fread((char *)(buffer),sizeof(short),bufferSize,input))!=0 ){
		std::cout<<"Read "<<length<<" words"<<std::endl;
		// Sliding buffer to be offset by the examiner after if swallows next DDU/DCC event from the block
		const unsigned short *buffer_end   = buffer;
		// Beginning of the block, which processed by the examiner (DDU/DCC/unidentified)
		const unsigned short *buffer_start = buffer;

		while( (length = examiner.check(buffer_end,length)) >= 0 ){
			std::cout<<"Event size="<<length<<std::endl;
			// If examiner is in DCC mode, several DDUs may be found in event
			//  (for DDU mode event always consists from one DDU)
			std::vector<int> sourceIDs = examiner.listOfDDUs();
			// Iterate over all DDUs from the event
			for(unsigned int ddu=0; ddu<sourceIDs.size(); ddu++){
				// Obtain pointer to the DDU data block
				const unsigned short* block = examiner.DDU_block()[sourceIDs[ddu]];
				// Obtain offset relative to 'buffer_start', which identifies DDU data block
				unsigned long offset = examiner.DDU_ptrOffsets()[sourceIDs[ddu]];
				// Example of accessing DDU data:
				std::cout<<"DDU "<<sourceIDs[ddu]<<std::hex
					<<" 0x"<<*(block+0)
					<<" 0x"<<*(block+1)
					<<" 0x"<<*(block+2)
					<<" 0x"<<*(block+3)
					<<" size="<<std::dec<<examiner.DDU_size()[sourceIDs[ddu]]<<std::endl<<" method2"<<std::hex
					<<" 0x"<<*(buffer_start+offset+0)
					<<" 0x"<<*(buffer_start+offset+1)
					<<" 0x"<<*(buffer_start+offset+2)
					<<" 0x"<<*(buffer_start+offset+3)
					<<std::dec<<std::endl;
				// Any errors for the DDU?
				if(examiner.errorsForDDU(sourceIDs[ddu]))
					std::cout<<std::hex<<"Errors for ddu=0x"<<sourceIDs[ddu]
						<<" : 0x"<<examiner.errorsForDDU(sourceIDs[ddu])<<std::dec<<std::endl;
				// Obtain pointer to a certain DMB data block
				std::map<short,const unsigned short*> DMBs = examiner.DMB_block()[sourceIDs[ddu]];
				// Obtain offset relative to 'buffer_start', which identifies certain DMB data block
				std::map<short,unsigned long> _DMBs = examiner.DMB_ptrOffsets()[sourceIDs[ddu]];
				// Example of accessing DMB data:
				for(std::map<short,const unsigned short*>::const_iterator dmb=DMBs.begin(); dmb!=DMBs.end(); dmb++)
					std::cout<<"  DMB "<<dmb->first<<std::hex
						<<" 0x"<<*(dmb->second+0)
						<<" 0x"<<*(dmb->second+1)
						<<" 0x"<<*(dmb->second+2)
						<<" 0x"<<*(dmb->second+3)
						<<" size="<<std::dec<<examiner.DMB_size()[sourceIDs[ddu]][dmb->first]<<std::endl;
				for(std::map<short,unsigned long>::const_iterator dmb=_DMBs.begin(); dmb!=_DMBs.end(); dmb++)
					std::cout<<" method2"<<std::hex
						<<" 0x"<<*(buffer_start+dmb->second+0)
						<<" 0x"<<*(buffer_start+dmb->second+1)
						<<" 0x"<<*(buffer_start+dmb->second+2)
						<<" 0x"<<*(buffer_start+dmb->second+3)
						<<std::dec<<std::endl;
			}
			buffer_start = buffer_end;
		}
	}

	return 0;
}
