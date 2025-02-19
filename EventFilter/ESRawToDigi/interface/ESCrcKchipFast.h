#ifndef ESCrcKchipFast_H
#define ESCrcKchipFast_H

class ESCrcKchipFast {

   private :

     uint32_t crc;

   public:

      ESCrcKchipFast() {
         init();
         reset() ;
      };

      void init() {
        crc = 0x0ffff ;
      }

      void reset() {
        crc = 0x0ffff ;
      } ;

      void add(unsigned int data) {
	for (int i=0;i<16;i++)
	  {
	    if ((crc&0x0001) == (data&0x0001))
	      crc=crc>>1;
	    else
	      crc=(crc>>1)^0x8408; // flipped 0x1021;
	    data=(data>>1);
	  }
      };
      
      uint32_t get_crc() {
        return crc ;
      };
      
      bool isCrcOk(unsigned int crcin=0x0000) {
          return ((get_crc()==crcin) ? true : false );
      };

      ~ESCrcKchipFast() { } ;

};

#endif
