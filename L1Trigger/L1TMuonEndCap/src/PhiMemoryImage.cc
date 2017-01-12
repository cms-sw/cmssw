#include <iostream>
#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"

PhiMemoryImage::PhiMemoryImage():_keyStationOffset(0){
  for (int i = 0; i < PhiMemoryImage::TOTAL_UNITS; i++) _buffer[i] = 0;
}

PhiMemoryImage::PhiMemoryImage(PhiMemoryImage::value_ptr buffer, int offset):_keyStationOffset(offset){

  CopyFromBuffer( buffer, offset );

}

void PhiMemoryImage::CopyFromBuffer(PhiMemoryImage::value_ptr rhs, int offset){

  _keyStationOffset = offset;

  for (int i = 0; i < PhiMemoryImage::TOTAL_UNITS; i++){
    _buffer[i] = rhs[i];
  
  }

}

//void SetBuff(int chunk, int value){

	//_buffer[chunk] = value;
//}


void PhiMemoryImage::SetBit( int station, int bitNumber ,bool bitValue ){

  //std::cout<<"bitnumber = "<<bitNumber<<std::endl;
 //  std::cout<<"station = "<<station<<std::endl;
  
  PhiMemoryImage::value_type tmp = 1;//64 bit word
  int size = sizeof (PhiMemoryImage::value_type) * 8;//64  // <- should be hard coded total units?

  if (bitNumber > size * PhiMemoryImage::TOTAL_UNITS){
    // complain in some way .. to be implemented..
	
	std::cout<<"bit number is greater than total size. Don't do that!\n";
    return;
  }

  station = station-1;
  bitNumber = bitNumber-1;
  
  
  // std::cout<<"bitnumber = "<<bitNumber<<std::endl;
   // std::cout<<"station = "<<station<<std::endl;
 
  int chunkNumber = station*PhiMemoryImage::UNITS + (bitNumber-1) / size;
  int bitOffset   = bitNumber % size;
  if(bitNumber == 64 || bitNumber == 128){chunkNumber += 1;}
  
  
  // std::cout<<"chunknumber = "<<chunkNumber<<std::endl;
  // std::cout<<"bitOffset = "<<bitOffset<<std::endl;

  //std::cout<<"tmp = "<<tmp<<std::endl;
  tmp = tmp << bitOffset;//
  //std::cout<<"tmp = "<<tmp<<std::endl;

  if (bitValue)
    _buffer[chunkNumber] |= tmp;
  else{
    tmp = ~tmp;
    _buffer[chunkNumber] &= tmp;
  }
  
  //std::cout<<"buffer["<<chunkNumber<<"] = "<<_buffer[chunkNumber]<<std::endl;
  //std::cout<<"buffer["<<chunkNumber + 1<<"] = "<<_buffer[chunkNumber+1]<<std::endl;

}

bool PhiMemoryImage::GetBit( int station, int bitNumber) const {

  PhiMemoryImage::value_type tmp = 1;
  int size = sizeof (PhiMemoryImage::value_type) * 8;// should be hardcoded total units?
  
  if (bitNumber > (size * PhiMemoryImage::TOTAL_UNITS)){
    // complain in some way .. to be implemented..
    return false;
  }

  bitNumber -= 1;
  station -=1;

  int chunkNumber = station*PhiMemoryImage::UNITS + ((bitNumber-1) / size);///changed this
  int bitOffset   = bitNumber % size;
  if(bitNumber == 64 || bitNumber == 128){chunkNumber += 1;}

  tmp <<= bitOffset;

  // std::cout << "chunkNumber: " << chunkNumber << "bitOffset:" 
  //     	    << bitOffset << "tmp: " << tmp << std::endl;
  //std::cout<<"buffer["<<chunkNumber<<"] = "<<_buffer[chunkNumber]<<std::endl;
 // std::cout<<"buffer["<<chunkNumber + 1<<"] = "<<_buffer[chunkNumber+1]<<std::endl;

  return ((_buffer[chunkNumber] & tmp) != 0);

}



void PhiMemoryImage::BitShift(int nBits){

  if (nBits == 0) return;

  bool negShift = (nBits < 0);

  if (negShift) nBits = -nBits;

  PhiMemoryImage::value_type transferBits, transferBits2;
  int value_size = sizeof(transferBits)*8;//should be hardcoded total units?

  //std::cout<<"value_size = "<<value_size<<"\n";

  for (int i = 0; i < PhiMemoryImage::STATIONS; i++){

    if (negShift){

      transferBits = (0x1 << nBits) - 1;
      transferBits &= _buffer[3*i+1];
	  
	  transferBits2 = (0x1 << nBits) - 1;
      transferBits2 &= _buffer[3*i+2];
	
	  _buffer[3*i+2] >>= nBits;
      _buffer[3*i+1] >>= nBits;
      _buffer[3*i]   >>= nBits;

      transferBits <<= (value_size - nBits);
	  transferBits2 <<= (value_size - nBits);

      _buffer[3*i] |= transferBits;
	  _buffer[3*i+1] |= transferBits2;

    } else {

      transferBits = (0x1 << nBits) - 1;
      transferBits <<= (value_size - nBits);
	  
	  transferBits2 = (0x1 << nBits) - 1;
      transferBits2 <<= (value_size - nBits);
	  
	 // if(!i){
	  
	  //	std::cout<<"tb = "<<transferBits<<"\n";
		//std::cout<<"tb2 = "<<transferBits2<<"\n";
	  
	 
	//  	std::cout<<"buf+0 = "<<_buffer[3*i]<<"\n";
	 // 	std::cout<<"buf+1 = "<<_buffer[3*i+1]<<"\n";
	 // 	std::cout<<"buf+2 = "<<_buffer[3*i+2]<<"\n";
	 // }

      transferBits &= _buffer[3*i];
      transferBits >>= (value_size - nBits);
	  
	  transferBits2 &= _buffer[3*i+1];
      transferBits2 >>= (value_size - nBits);
	  
	  //if(!i){
	 // 	std::cout<<"tb = "<<transferBits<<"\n";
	 // 	std::cout<<"tb2 = "<<transferBits2<<"\n";
	 // }
      _buffer[3*i] <<= nBits;
      _buffer[3*i+1] <<= nBits;
	  _buffer[3*i+2] <<= nBits;
	  
	 // if(!i){
	 // 	std::cout<<"buf+0 = "<<_buffer[3*i]<<"\n";
	  //	std::cout<<"buf+1 = "<<_buffer[3*i+1]<<"\n";
	 // 	std::cout<<"buf+2 = "<<_buffer[3*i+2]<<"\n";
	 // }
      
      _buffer[3*i+1] |= transferBits;
	  _buffer[3*i+2] |= transferBits2;
	  
	 // if(!i){
	    
	//	std::cout<<"tb = "<<transferBits<<"\n";
	  //	std::cout<<"tb2 = "<<transferBits2<<"\n";
	  //  
	  //	std::cout<<"buf+0 = "<<_buffer[3*i]<<"\n";
	  //	std::cout<<"buf+1 = "<<_buffer[3*i+1]<<"\n";
	 // 	std::cout<<"buf+2 = "<<_buffer[3*i+2]<<"\n";
	  //}
    }
  }

}

void PhiMemoryImage::Print(){

  int size = PhiMemoryImage::UNITS * sizeof(PhiMemoryImage::value_type)*8;//should be hardcoded total units>?

  for (int i = 1; i <= PhiMemoryImage::STATIONS; i++){

    //    std::cout << _buffer[(i-1)*2] << " " << _buffer [(i-1)*2+1] << std::endl;

    for (int j = size; j > 0; j--){

      if ((j%(sizeof(PhiMemoryImage::value_type)*8)) == 0)//should be hardcoded total units>?  ->no
	std::cout << std::endl;
    
      
      if ((j%8)==0) std::cout << " ";
      if(GetBit(i,j)) std::cout << "1";
      else std::cout << "0";
      
    }
    std::cout << std::endl;
  }

}

void PhiMemoryImage::printbuff(){

	for(int i=0;i<PhiMemoryImage::TOTAL_UNITS;i++){
		std::cout<<"buffer["<<i<<"] = "<<_buffer[i]<<std::endl;
	}
	std::cout<<std::endl;
}


