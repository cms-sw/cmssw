#ifndef FEDRawData_DaqData_h
#define FEDRawData_DaqData_h

/**  \class DaqData
 *  
 *  This is the basic class accomplishing raw data 
 *  formatting/unformatting.
 *
 *  
 *  1) Creation of a buffer of bytes representing the raw data 
 *
 *  the user must create a DaqData object providing as input the data 
 *  in the form of a vector of unsigned integers. The input data 
 *  are all the values necessary to specify one (or more) object of the given
 *  format, which is the templated type. In case more than one object 
 *  of the same given format should be put in the buffer, then the values 
 *  of the second object must follow in the vector those of the first 
 *  object and so on. DaqData will then compress these values in a bitstream 
 *  according to the desired format. The bitstream is always aligned 
 *  to the byte. 
 *  The user, after successful compression, can request the pointer to the 
 *  beginning of the buffer of bytes that represent the raw data 
 *  (method Buffer() ) and the length of this buffer (method Size() ).  
 *
 *
 *  2) Interpretation of a buffer of bytes representing the raw data
 *
 *  the user must create a DaqData object providing as input a pointer 
 *  to a buffer of bytes and the length in bytes of this buffer. DaqData will 
 *  then extract the values (unsigned integers) present in the buffer 
 *  according to a proper format, which is again the templated type.
 *  After successful uncompression of the buffer, the user can request 
 *  the number of objects of the given format that were present in the buffer 
 *  ( method Nobjects() ) and retrieve the individual values characterizing 
 *  these objects ( method getValue() ).
 *
 *  WARNING: at the moment the interpretation of a buffer of bytes proceeds
 *           through the constructor of DaqData. This operation implies more 
 *           than one copy of the data, which is not the most efficient way 
 *           of accomplishing the task. In a future version of DaqData, 
 *           more efficient methods will be made available.
 *
 *  $Date: 2005/10/06 18:25:22 $
 *  $Revision: 1.2 $
 *  \author G. Bruno  - CERN, EP Division
 */


#include <typeinfo>  
#include <vector>  
#include <map>
#include <string>
#include <cmath>
   
template <class Format>
class DaqData {

 public:

  DaqData(std::vector<unsigned int> & v) : size_(0), buffer_(0), nobjects_(0) {

    try {

      if (v.size() == 0)  throw string("DaqData: empty input data vector provided: ");     

      int Nfields = Format::getNumberOfFields();
      if (v.size()%Nfields != 0)  throw string("DaqData: ERROR. You must provide a number of input values compatibles with the requested format: ");


      int ObjSize = Format::getFieldLastBit(Nfields-1)+1;
      nobjects_ = v.size()/Nfields;
      //      cout<<"Going to create "<<nobjects_<< " objects of type "<<typeid(Format).name()<<endl;

      
      size_ = (int) ceil(((double)(ObjSize*nobjects_))/8);
      buffer_= new  char[size_];      
      //   cout<<"The buffer will be "<<size_ <<" bytes long"<<endl; 

      std::vector<unsigned int>::iterator vit = v.begin();
      char * p = buffer_;
      int bitsalreadyfilled = 0;

      int Totbytes=1;

      for(int i=0; i<nobjects_; i++) {

	//additional bytes necessary to accomodate the current object
	int Nbytes = (int) ceil((double)(Format::getFieldLastBit(Nfields-1)+1-8+bitsalreadyfilled)/8); 

	if ((Totbytes+=Nbytes)  > size_) throw string("Exceeded allocated buffer size");
	
	compressObject(p, vit, bitsalreadyfilled);

       	// cout<<"Successfully compressed object "<< i <<endl; 
	  
	data_.insert(std::pair< int, std::vector<unsigned int> >(i,std::vector<unsigned int>(vit-Nfields,vit) ));

	//the second term is necessary in the case the last byte has been fully used 
	p+=(Nbytes+(8-bitsalreadyfilled)/8) ;
	Totbytes+=(8-bitsalreadyfilled)/8 ;
      }
      if(bitsalreadyfilled==0) Totbytes--;

      /*
       cout << "Compression successful. "<< Totbytes<<" bytes compressed in total"<< endl;
       cout<< "Buffer pointer: "<<hex<< buffer_<< dec << "Size of buffer= "<< size_<<endl;
      for(int i=0; i<nobjects_; i++) {
	cout << "Object # "<< i << " fields: " <<endl;
	for(int j=0; j<Format::getNumberOfFields(); j++) cout << getValue(j,i) <<endl;
      }
      */

    }

    catch (std::string s){

      cout<<"DaqData - Exception caught: " << s <<endl;
      cout<<"Object compression failed! No data has been constructed for format: "<< string(typeid(Format).name())<<endl; 
      
    }

  }


  DaqData(const unsigned char * ptr, 
	  int sizeinbytes) : size_(0), buffer_(0), nobjects_(0) {


    try {

      if (sizeinbytes==0) throw std::string("Buffer size is zero");

      int Nfields = Format::getNumberOfFields();
      int ObjSize = Format::getFieldLastBit(Nfields-1)+1;
      nobjects_ = (sizeinbytes*8)/ObjSize;

      // cout<<"Going to create "<<nobjects_<< " objects of type "<<typeid(Format).name()<<endl;
      // cout<<"The buffer will be "<<size_ <<" bytes long"<<endl;      

      if ((sizeinbytes*8)%ObjSize != 0) {
	cout<<"DaqData: there will be  " << (sizeinbytes*8)%ObjSize <<" meaningless bits at the end of the buffer"<<endl;
      }

      //     buffer_ = new char[sizeinbytes];
      //     memmove(buffer_,ptr,sizeinbytes);

      //      char * p = buffer_;

      const unsigned char * p = ptr;
      int bitsalreadyfilled = 0;
      
      int Totbytes=1;

      for(int i=0; i<nobjects_; i++) {

	std::vector<unsigned int> objdata;
	objdata.reserve(Nfields);

	//additional bytes necessary to accomodate the current object
	int Nbytes = (int) ceil((double)(Format::getFieldLastBit(Nfields-1)+1-8+bitsalreadyfilled)/8); 


	if ((Totbytes+=Nbytes) > sizeinbytes) throw std::string("Exceeded allocated buffer size");

	
	uncompressObject(p, objdata, bitsalreadyfilled);
     
	//	cout<<"Successfully uncompressed object "<< i <<endl; 
	data_.insert(std::pair< int, std::vector<unsigned int> >(i,objdata) );

	//the second term is necessary in the case the last byte has been fully used 
	p+= (Nbytes + (8-bitsalreadyfilled)/8);
	Totbytes+= (8-bitsalreadyfilled)/8;

      }

      if(bitsalreadyfilled==0) Totbytes--;

      /*
      cout << "Uncompression succeeded. "<< Totbytes<<" bytes uncompressed in total"<< endl;
      cout<< "Buffer pointer: "<<hex<< buffer_<< dec<< "Size of buffer= "<< size_<<endl;

      for(int i=0; i<nobjects_; i++) {
	cout << "Object # "<< i << " fields: " <<endl;
	for(int j=0; j<Format::getNumberOfFields(); j++) cout << getValue(j,i) <<endl;
      }
      */

    }
    catch (std::string s){

      std::cout<<"DaqData - Exception caught: " << s <<std::endl;
      std::cout<<"Object uncompression failed! No data has been constructed for format: "<<std::string(typeid(Format).name())<<endl; 
      
    }

  }


  DaqData(const DaqData & a) {

    size_ = a.Size();
    buffer_ = new char[size_];
    memmove(buffer_,a.Buffer(),size_);    
    nobjects_ = a.Nobjects();

    for(int i=0; i<nobjects_; i++) {
      std::vector<unsigned int> vec;
      for(int j=0; j<Format::getNumberOfFields(); j++) vec.push_back(a.getValue(j,i)); 
      data_.insert(std::pair< int, std::vector<unsigned int> >(i, vec));
    }

  }


  ~DaqData(){

    // cout<<"Deleting DaqData of type "<<typeid(Format).name()<<endl;
    if (buffer_!=0) delete [] buffer_;

  };


  char * Buffer() const {return buffer_;}

  int Size() const {return size_;}

  int Nobjects() const {
    return nobjects_;
    /*    cout<<"Nobjects()"<<endl;
    int Nfields = Format::getNumberOfFields();
    return size_/(Format::getFieldLastBit(Nfields-1)+1);
    */
  }

  unsigned int getValue(int indfield, int indobj=0) const {
    
    if (indobj<nobjects_ && indfield < Format::getNumberOfFields()) {
      std::map < int, std::vector<unsigned int> >::const_iterator it = data_.find(indobj);
      if (it != data_.end()) return ((*it).second)[indfield];
      else {
	cout<<"DaqData - Strange: object should exist but was not found "<<endl;
	return 0;
      }
    }
    else  cout<<"DaqData - Non existent field or object"<<endl;
    return 0;

  }

 private:

  void uncompressObject(const  unsigned char * ptr, 
			std::vector<unsigned int> & objdata, 
			int & bitsalreadyfilled) {

    int Nfields = Format::getNumberOfFields();

    int bitstoaccomodate=Format::getFieldLastBit(0)+1; // of current field

    int ifield = 0;
    unsigned int value = 0; 

    while (ifield < Nfields) {

      if(bitstoaccomodate > 8 - bitsalreadyfilled ) {

       	// cout<<"can't complete value from current byte"<<endl;
	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

	//1)The syntax below could be faster. 
	//  To be checked as soon as time available(check started with prog test4.C) .
	//2)check if all cast to unsigned int are really necessary(done, they are not).
	//3)Instead of using pow(2,1;2;3;4;5..), a faster enum could be used
	//Lower all bits but those not yet read
	//value += ( (*ptr & ((unsigned int)pow(2.,8-bitsalreadyfilled)-1)) << bitstoaccomodate + bitsalreadyfilled - 8 ) ;
	//       	if(bitstoaccomodate > 8) value += ( ((((unsigned int)(*ptr)) << bitsalreadyfilled) & 0xff) << (bitstoaccomodate - 8) );
	// 	else value += ( ((((unsigned int)(*ptr)) << bitsalreadyfilled) & 0xff) >> (8 - bitstoaccomodate) );


	if(bitstoaccomodate > 8) value += ( ( (*ptr << bitsalreadyfilled) & 0xff ) << (bitstoaccomodate - 8) );
       	else value += ( ( (*ptr << bitsalreadyfilled) & 0xff) >> (8 - bitstoaccomodate) );
	

       	// cout<< "value: "<< hex << value << " - " << dec <<  value<< endl;
       	// cout<< "byte: "<< hex << (unsigned int)(*ptr) << " - " << dec << (unsigned int)(*ptr) << endl;


	ptr++;
	bitstoaccomodate -= (8-bitsalreadyfilled);
	bitsalreadyfilled = 0;

      	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;
	
      } 

      else if(bitstoaccomodate < (8 - bitsalreadyfilled) && bitstoaccomodate >0){
       	// cout<<"value can be completed from current byte, which still contain info"<<endl;

       	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

	//1)The syntax below could be faster. 
	//  To be checked as soon as time available.
	//2)check if all cast to unsigned int are really necessary.
	//3)Instead of using pow(2,1;2;3;4;5..), a faster enum could be used
	//Lower all bits but those not yet read
	//value += (*ptr & ((unsigned int)pow(2.,8-bitsalreadyfilled)-1)) >> (8 - bitstoaccomodate - bitsalreadyfilled);

       	value += ( ( (*ptr << bitsalreadyfilled) & 0xff ) >> (8 - bitstoaccomodate) ) ;

       	// cout<< "value: "<< hex << value << " - " << dec <<  value<< endl;
       	// cout<< "byte: "<< hex << (unsigned int)(*ptr) << " - " << dec << (unsigned int)(*ptr) << endl;


	objdata.push_back(value);
	value = 0;
	bitsalreadyfilled+=bitstoaccomodate;

       	// cout<<"Field completed"<<endl;

	if(ifield==Nfields-1) return;
       
       	// cout<<"Uncompressing new Field"<<endl;

	ifield++;
	bitstoaccomodate=Format::getFieldLastBit(ifield)-Format::getFieldLastBit(ifield-1); 
	
       	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

      }

      else if(bitstoaccomodate == (8 - bitsalreadyfilled) && bitstoaccomodate >0){
       	// cout<<"value can be completed from what left in current byte"<<endl; 
       	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

	//1)The syntax below could be faster. 
	//  To be checked as soon as time available.
	//2)check if all cast to unsigned int are really necessary.
	//3)Instead of using pow(2,1;2;3;4;5..), a faster enum could be used
	//Lower all bits but those not yet read
	//	value += *ptr & ((unsigned int)pow(2.,8-bitsalreadyfilled)-1);

       	value += ( ( (*ptr << bitsalreadyfilled) & 0xff ) >> (8 - bitstoaccomodate) ) ;

       	// cout<< "value: "<< hex << value << " - " << dec <<  value<< endl;
       	// cout<< "byte: "<< hex << (unsigned int)(*ptr) << " - " << dec << (unsigned int)(*ptr) << endl;


	objdata.push_back(value);
	value = 0;
	bitsalreadyfilled=0;

       	// cout<<"Field completed"<<endl;

	if(ifield==Nfields-1) return;

       	// cout<<"Uncompressing new Field"<<endl;

	ptr++;
	ifield++;
	bitstoaccomodate=Format::getFieldLastBit(ifield)-Format::getFieldLastBit(ifield-1); 	

       	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

      }
      else throw std::string(" unexpected situation during uncompression");

    } //end of cycle over fields
    
  }




  void compressObject(const unsigned char * ptr, std::vector<unsigned int>::iterator & vit, int & bitsalreadyfilled) {

    int Nfields = Format::getNumberOfFields();

    int bitstoaccomodate=Format::getFieldLastBit(0)+1; // of current field
    
    if (*vit > pow(2.,bitstoaccomodate)-1) throw string("The value is too large to fit in the field ");
 
    int ifield = 0;

    //Lower all bits but those already filled
    *ptr &= 0xff + 1 - (unsigned int)pow(2.,8-bitsalreadyfilled);

    while (ifield < Nfields) {


      if(bitstoaccomodate > 8 - bitsalreadyfilled ) {

       	// cout<<"Field cannot be compressed from what left in current byte"<<endl;
	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

	*ptr += (((*vit) >> (bitstoaccomodate - (8 - bitsalreadyfilled))) & 0xff);
	// cout<< "value: "<< hex << *vit <<" - " << dec <<  *vit<< endl;
	// cout<< "byte: "<< hex << (unsigned int)(*ptr) <<" - "<< dec << (unsigned int)(*ptr) << endl;


	bitstoaccomodate -= (8-bitsalreadyfilled);
	bitsalreadyfilled = 0;
	ptr++;
	*ptr &= 0xff + 1 - (unsigned int)pow(2.,8-bitsalreadyfilled);

	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

      } 

      else if(bitstoaccomodate < (8 - bitsalreadyfilled) && bitstoaccomodate >0){

	// cout<<"Field can be compressed in the current byte, which will not be completely filled"<<endl;

	*ptr += ( (((*vit) << 8-bitstoaccomodate) & 0xff) >> (bitsalreadyfilled) );

	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;
	// cout<< "value: "<< hex << *vit <<" - " << dec <<  *vit<< endl;
	// cout<< "byte: "<< hex << (unsigned int)(*ptr) <<" - "<< dec << (unsigned int)(*ptr) << endl;

	vit++;
	bitsalreadyfilled+=bitstoaccomodate;

	// cout<<"Field completed"<<endl;

	if(ifield==Nfields-1) return;

	// cout<<"Compressing new Field"<<endl;
       
	ifield++;
	bitstoaccomodate=Format::getFieldLastBit(ifield)-Format::getFieldLastBit(ifield-1); 
	if (*vit > pow(2.,bitstoaccomodate)-1) throw string("The value is too large to fit in the field ");

	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

      }

      else if(bitstoaccomodate == (8 - bitsalreadyfilled) && bitstoaccomodate >0){

	// cout<<"Field can be compressed in the current byte, which will be completely filled"<<endl;


	*ptr += ( (((*vit) << 8-bitstoaccomodate) & 0xff) >> (bitsalreadyfilled) );

	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;
	// cout<< "value: "<< hex << *vit <<" - " << dec <<  *vit<< endl;
	// cout<< "byte: "<< hex << (unsigned int)(*ptr) <<" - "<< dec << (unsigned int)(*ptr) << endl;

	vit++;
	bitsalreadyfilled=0;

	// cout<<"Field completed"<<endl;

	if(ifield==Nfields-1) return;

	// cout<<"Compressing new Field"<<endl;

	ptr++;
	*ptr &= 0xff + 1 - (unsigned int)pow(2.,8-bitsalreadyfilled);

	ifield++;
	bitstoaccomodate=Format::getFieldLastBit(ifield)-Format::getFieldLastBit(ifield-1); 	

	if (*vit > pow(2.,bitstoaccomodate)-1) throw string("The value is too large to fit in the field ");

	// cout <<"bitstoaccomodate= "<<bitstoaccomodate<<" bitsalreadyfilled= "<<bitsalreadyfilled<<" ifield="<< ifield<<" Nfields="<<Nfields<<endl;

      }
      else throw string(" unexpected situation during compression");

    } //end of cycle over fields
    
  }


 private:

  std::map < int, std::vector<unsigned int> > data_;

  int size_;
  unsigned char * buffer_;
  int nobjects_;

  /* Old implementation of compress
 
  //assumes a correct number of objects has been sent. Alignment to the byte
  int compressObject(unsigned char * p, vector<unsigned int>::iterator & vit, int & bitsalreadyfilled){
 
    int Nfields = Format::getNumberOfFields();
    int Nbytes = (int) (ceil((double)(Format::getFieldLastBit(Nfields-1)+1)/8));
 
    unsigned int indfield=0; // index of current field
    unsigned int bitstoaccomodate=Format::getFieldLastBit(indfield)+1; // of current field

    //Here there should be a check on the size of the first value against the available bits

    for (int ibyte = 0; ibyte < Nbytes ; ibyte++){

      //      int bitsalreadyfilled = 0; // of current byte

      if(bitstoaccomodate>8-bitsalreadyfilled) {
	cout<<"More than available bits to accomodate"<<endl;

	*p = ( (*vit) >> (bitstoaccomodate - 8) ) & 0xff;
	cout<< "value: "<< hex << *vit << dec <<  *vit<< endl;
	cout<< "byte: "<< hex << *p << dec << *p << endl;

	bitstoaccomodate -= 8;
	p++;
	bitsalreadyfilled = 0;     
      } 

      else if(bitstoaccomodate<=8-bitsalreadyfilled && bitstoaccomodate>0){

	cout<<"bits to accomodate less than available"<<endl;

	unsigned int bytevalue=0;

	// First, put the bits of the current object in the current byte
	
	bytevalue += ( (*vit) << ( 8 - bitstoaccomodate ) ) & 0xff;
	cout<< "value: "<< hex << *vit << dec <<  *vit<< endl;
	cout<< "byte: "<< hex << bytevalue << dec << bytevalue << endl;


	bitsalreadyfilled = bitstoaccomodate;
	indfield++;

	if (indfield>=Nfields)  return 0;
       
	vit++;

	//Here there should be a check on the size of the current value against the left available bits

	if(bitsalreadyfilled==8) {
	  bitstoaccomodate=Format::getFieldLastBit(indfield)-Format::getFieldLastBit(indfield-1);
	  p++;
	  continue;
	}

	//This is the case of remaining fields and space left in the byte
      
	//Compute the last field that can be treated for this byte
	int lastfield=indfield;
	int ntotbits=bitsalreadyfilled+Format::getFieldLastBit(indfield)-Format::getFieldLastBit(indfield-1);
	while(ntotbits < 8 && lastfield<Nfields )  {
	  lastfield++;
	  ntotbits+=Format::getFieldLastBit(lastfield)-Format::getFieldLastBit(lastfield-1);
	}

	// First, fit fields, if there are, that for sure fully fit into the current byte leaving at least one free bit
	for(; indfield < lastfield  ; indfield++) {
	  bitsalreadyfilled += Format::getFieldLastBit(indfield)-Format::getFieldLastBit(indfield-1);
	  
	  bytevalue += ((*vit)<<(8-bitsalreadyfilled)) ;
	  cout<< "value: "<< hex << *vit << dec <<  *vit<< endl;
	  cout<< "byte: "<< hex << bytevalue << dec << bytevalue << endl;
	  
	  if(indfield<Nfields-1) {
	    vit++;
	    //Here there should be a check on the size of the current value against the left available bits.
	  }
	  else return 0;
	}

	//Special treatment of last field having at least some space in the byte. At this point it should always be indfield==lastfield 
	
	int lastfieldbits=Format::getFieldLastBit(indfield)-Format::getFieldLastBit(indfield-1);
	bitstoaccomodate = lastfieldbits - (8-bitsalreadyfilled);
	
	bytevalue += ((*vit) >> (lastfieldbits - (8-bitsalreadyfilled))) ; 
	cout<< "value: "<< hex << *vit << dec <<  *vit<< endl;
	cout<< "byte: "<< hex << bytevalue << dec << bytevalue << endl;
	
	*p=bytevalue;
	
	//Treatment of the case in which the last field fits too in the current byte
	if (bitstoaccomodate==0) {

	  if (indfield < Nfields-1) {

	    indfield++;
	    vit++;

	    //Here there should be a check on the size of the current value against the left available bits.
	    
	    bitstoaccomodate=Format::getFieldLastBit(indfield)-Format::getFieldLastBit(indfield-1);
	  }
	  else return 0;
	}
	p++;
      }
      else{
	cout<<"DaqData - ERROR: unexpected situation"<<endl;
      }
    } //end of cycle over bytes
  
    return 0;

  }

*/



};


#endif
