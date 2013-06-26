#include "DataFormats/L1CaloTrigger/interface/L1CaloMipQuietRegion.h"


// Namespace resolution
using std::ostream;
using std::endl;
using std::hex;
using std::dec;
using std::showbase;
using std::noshowbase;


L1CaloMipQuietRegion::L1CaloMipQuietRegion():
  m_id(),
  m_data(0),
  m_bx(0)
{
}

L1CaloMipQuietRegion::L1CaloMipQuietRegion(bool mip, bool quiet, unsigned crate,
                                           unsigned card, unsigned rgn, int16_t bx):
  m_id(crate, card, rgn),
  m_data(0), // Over-ridden below
  m_bx(bx)
{
  pack(mip, quiet);
}

L1CaloMipQuietRegion::L1CaloMipQuietRegion(bool mip, bool quiet, unsigned ieta,
                                           unsigned iphi, int16_t bx):
  m_id(ieta, iphi),
  m_data(0), // Over-ridden below
  m_bx(bx)
{
  pack(mip, quiet);
}

bool L1CaloMipQuietRegion::operator==(const L1CaloMipQuietRegion& rhs) const
{
  return ( m_data==rhs.raw() && m_bx==rhs.bx() && m_id==rhs.id() );
}

ostream& operator<< (ostream& os, const L1CaloMipQuietRegion& rhs)
{
  os <<"L1CaloMipQuietRegion:"
     << " MIP=" << rhs.mip()
     << " Quiet=" << rhs.quiet() << endl
     << " RCT crate=" << rhs.rctCrate()
     << " RCT card=" << rhs.rctCard()
     << " RCT rgn=" << rhs.rctRegionIndex()
     << " RCT eta=" << rhs.rctEta()
     << " RCT phi=" << rhs.rctPhi() << endl
     << " GCT eta=" << rhs.gctEta()
     << " GCT phi=" << rhs.gctPhi()
     << " BX=" << rhs.bx() << endl;
  return os;
}
