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

	const    int   bufferSize = 116384;
	unsigned short buffer[bufferSize];
	int length = 0;
	bzero(buffer,sizeof(buffer));

	while( !feof(input) && (length=fread((char *)(buffer),sizeof(short),bufferSize,input))!=0 ){
		std::cout<<"Read "<<length<<" words"<<std::endl;
		const unsigned short *buf = buffer;

		while( (length = examiner.check(buf,length)) >= 0 ){
			std::cout<<"Length="<<length<<std::endl;
			std::vector<int> sourceIDs = examiner.listOfDDUs();
			for(int ddu=0; ddu<sourceIDs.size(); ddu++){
				const unsigned short* block = examiner.DDU_block()[sourceIDs[ddu]];
				std::cout<<"DDU "<<sourceIDs[ddu]<<std::hex<<" 0x"<<*(block+0)<<" 0x"<<*(block+1)<<" 0x"<<*(block+2)<<" 0x"<<*(block+3)<<std::dec<<std::endl;
				if(examiner.errorsForDDU(sourceIDs[ddu]))
					std::cout<<std::hex<<"Errors for ddu=0x"<<sourceIDs[ddu]<<" : 0x"
						<<examiner.errorsForDDU(sourceIDs[ddu])<<std::dec<<std::endl;
				std::map<short,const unsigned short*> DMBs = examiner.DMB_block()[sourceIDs[ddu]];
				for(std::map<short,const unsigned short*>::const_iterator dmb=DMBs.begin(); dmb!=DMBs.end(); dmb++)
					std::cout<<"  DMB "<<dmb->first<<std::hex<<" 0x"<<*(dmb->second+0)<<" 0x"<<*(dmb->second+1)<<" 0x"<<*(dmb->second+2)<<" 0x"<<*(dmb->second+3)<<std::endl;
			}
		}
	}

	return 0;
}
