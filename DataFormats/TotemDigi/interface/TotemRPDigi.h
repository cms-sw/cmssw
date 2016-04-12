/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#ifndef DataFormats_TotemDigi_TotemRPDigi
#define DataFormats_TotemDigi_TotemRPDigi

class TotemRPDigi
{
  public:
    TotemRPDigi(unsigned short strip_no=0) : strip_no_(strip_no)
    {
    };

    unsigned short getStripNumber() const
    {
      return strip_no_;
    }
  
  private:
    unsigned short strip_no_;
};


inline bool operator< (const TotemRPDigi& one, const TotemRPDigi& other)
{
  return one.getStripNumber() < other.getStripNumber();
}

#endif
