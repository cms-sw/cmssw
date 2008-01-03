#include <stdio.h>
#include <errno.h>
#include "../src/FileReaderDDU.cc"
// To compile: g++ -o stripDCC stripDCC.cc

int main(int argc, char *argv[]){
	if(argc!=3){
		printf("Usage: ./stripDCC INPUT_FILENAME OUTPUT_FILENAME\n");
		return 0;
	}
	FILE *out;
	if( (out=fopen(argv[2],"wt"))==NULL ){
		printf("Cannot open output file: %s (errno=%d)\n",argv[2],errno);
		return 1;
	}
	FileReaderDDU reader;
	reader.reject(FileReaderDDU::Unknown);
	reader.select(0);
	try {
		reader.open(argv[1]);
	} catch ( std::runtime_error &e ){
		printf("Cannot open input file: %s (errno=%d)\n",argv[1],errno);
	}
	const unsigned short *buf=0;
	size_t length=0;
	int nEvents=0;
	while( (length=reader.next(buf))!=0 ){ fwrite(buf,2,length,out); nEvents++; }
	fclose(out);

	printf("nEvents=%d\n",nEvents);
	return 0;
}
