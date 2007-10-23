   #include "DQM/SiStripMonitorHardware/interface/Fed9UDebugEvent.hh"

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_8_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+8, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_8_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+12, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_8(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+16, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getBESR(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+18, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_8_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+22, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_7_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+24, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_7_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+28, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_7(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+32, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getRES_5(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+34, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_7_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+38, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_6_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+40, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_6_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+44, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_6(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+48, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getRES_4(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+50, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_6_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+54, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_5_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+56, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_5_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+60, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_5(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+64, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getRES_3(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+66, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_5_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+70, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_4_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+72, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_4_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+76, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_4(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+80, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getRES_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+82, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_4_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+86, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_3_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+88, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_3_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+92, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+96, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getRES_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+98, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_3_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+102, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_2_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+104, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_2_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+108, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_2(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+112, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getDAQ_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+114, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_2_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+118, true) & 0xFFFF);
   }

   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_1_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+120, true) & 0xFFFFFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getFSOP_1_2(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+124, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFLEN_1(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+128, true) & 0xFFFF);
   }
   Fed9U::u32 Fed9U::Fed9UDebugEvent::getDAQ_1(void) const {
      return (d_buffer.getu32(d_SPECIAL_OFF+130, true) & 0xFFFFFFFF);
   }
   Fed9U::u16 Fed9U::Fed9UDebugEvent::getFSOP_1_3(void) const {
      return (d_buffer.getu16(d_SPECIAL_OFF+134, true) & 0xFFFF);
   }
