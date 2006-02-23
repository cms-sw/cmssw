#ifndef CondFormats_EcalObjects_EcalXtalGroupId_H
#define CondFormats_EcalObjects_EcalXtalGroupId_H
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

class EcalXtalGroupId {
  public:
    EcalXtalGroupId();
    EcalXtalGroupId(const unsigned int& id);
    virtual ~EcalXtalGroupId();
    bool operator>(const EcalXtalGroupId& rhs) const{ return ( id_>rhs.id() ); }
    bool operator>=(const EcalXtalGroupId& rhs) const { return ( id_>=rhs.id() ); }
    bool operator==(const EcalXtalGroupId& rhs) const { return ( id_==rhs.id() ); }
    bool operator<(const EcalXtalGroupId& rhs) const { return ( id_<rhs.id() ); }
    bool operator<=(const EcalXtalGroupId& rhs) const { return ( id_<=rhs.id() ); }
    //EcalXtalGroupId& operator=(const EcalXtalGroupId& rhs) { return EcalXtalGroupId(rhs); }
    void operator=(const EcalXtalGroupId& rhs) { id_ = rhs.id_; }

    const unsigned int id() const { return id_; }
  private:
    unsigned int id_;

};
#endif
