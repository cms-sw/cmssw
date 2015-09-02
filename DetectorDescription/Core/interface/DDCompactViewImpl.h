#ifndef DDCompactViewImpl_h
# define DDCompactViewImpl_h

# include "DetectorDescription/Core/interface/DDLogicalPart.h"
# include "DetectorDescription/Core/interface/DDPosData.h"
# include "DetectorDescription/Core/interface/graphwalker.h"

class DDPartSelector;

class DDCompactViewImpl 
{
public:
  
  typedef ::graph<DDLogicalPart, DDPosData* > GraphNav;
  typedef graphwalker<DDLogicalPart, DDPosData* > WalkerType;

  explicit DDCompactViewImpl();
  DDCompactViewImpl(const DDLogicalPart & rootnodedata);
  ~DDCompactViewImpl();
  
  //reassigns root with no check!!!
  void setRoot(const DDLogicalPart & root) { root_=root; }

  const DDLogicalPart & root() const { return root_; }
  
  DDLogicalPart & current() const;
  
  const GraphNav & graph() const { return graph_; }

  graphwalker<DDLogicalPart,DDPosData*> walker() const; 
  
  double weight(const DDLogicalPart &) const;

  void position (const DDLogicalPart & self,
		 const DDLogicalPart & parent,
		 int copyno,
		 const DDTranslation & trans,
		 const DDRotation & rot,
		 const DDDivision * div);

  void swap( DDCompactViewImpl& );  

protected:    
  // internal use ! (see comments in DDCompactView(bool)!)
  DDLogicalPart root_;
  GraphNav graph_;
};

#endif
