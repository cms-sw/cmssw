/****************************************************************************
*
* Authors:
*	Jan Kašpar (jan.kaspar@gmail.com) 
*	CMSSW developpers (based on class GeometricDet)
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_DetGeomDesc
#define Geometry_VeryForwardGeometryBuilder_DetGeomDesc

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/DetId/interface/DetId.h"

class DDFilteredView;
class RPAlignmentCorrection;


/**
 * \brief Geometrical description of a detector.
 *
 * See schema of \ref TotemRPGeometry "TOTEM RP geometry classes"
 *
 * Class resembling GeometricDet class. Slight changes were made to suit needs of the TOTEM RP description.
 * Each instance is a tree node, with geometrical information from DDD (shift, rotation, material, ...), ID and list of children nodes.
 * It is intended to have two such a trees. One for ideal geometry (within IdealGeometryRecord) and second for real geometry (VeryForwardRealGeometryRecord).
 * The transition from ideal to real geometry (i.e. loading alignments) is done by TotemRPRealGeometryModule.
 * 
 * The <b>translation</b> and <b>rotation</b> parameters are defined by <b>local-to-global</b>
 * coordinate transform. That is, if r_l is a point in local coordinate system and x_g in global,
 * then the transform reads:
 \verbatim
    x_g = rotation * x_l + translation
 \endverbatim
 **/

class DetGeomDesc
{
	public:
		typedef std::vector< const DetGeomDesc*>  ConstContainer;
		typedef std::vector< DetGeomDesc*>  Container;
		typedef DDExpandedView::nav_type nav_type;
		
		/// a type (not used in the moment, left for the future)
		typedef unsigned int GeometricEnumType;
		
		///Constructors to be used when looping over DDD
		DetGeomDesc(nav_type navtype, GeometricEnumType dd = 0);
		DetGeomDesc(DDExpandedView* ev, GeometricEnumType dd = 0);
		DetGeomDesc(DDFilteredView* fv, GeometricEnumType dd = 0);
		
		/// copy constructor and assignment operator
		DetGeomDesc(const DetGeomDesc &);
		DetGeomDesc& operator= (const DetGeomDesc &);

		/// destructor
		virtual ~DetGeomDesc();
		
		/// ID stuff
		void setGeographicalID(DetId id) { _geographicalID = id; }
		virtual DetId geographicalID() const { return _geographicalID; }

		/// access to the tree structure
		virtual ConstContainer components() const;
		virtual Container components();
		virtual ConstContainer deepComponents() const;				/// returns all the components below
		virtual std::vector< DDExpandedNode > parents() const		/// retuns the geometrical history
			{ return _parents; }

		/// components (children) management
		void setComponents(Container cont)
			{ _container = cont; }
		void addComponents(Container cont);
		void addComponent(DetGeomDesc*);
		void clearComponents()
			{ _container.resize(0);} 
		void deleteComponents(); 									/// deletes just the first daughters
		void deepDeleteComponents();  								///traverses the treee and deletes all nodes.
		bool isLeaf() const 
			{ return (_container.size() == 0); }
		
		/// geometry information
		DDRotationMatrix	rotation() const {return _rot;}
		DDTranslation		translation() const {return _trans;}
		DDSolidShape		shape() const  {return _shape;}
		GeometricEnumType	type() const {return _type;}
		DDName				name() const {return _ddname;};
		nav_type			navType() const {return _ddd;}
		std::vector<double>	params() const {return _params;}
		virtual int			copyno() const {return _copy;}
		virtual double		volume() const {return _volume;}
		virtual double		density() const {return _density;}
		virtual double		weight() const {return _weight;}
		virtual std::string	material() const {return _material;}

		/// alignment
		void ApplyAlignment(const RPAlignmentCorrection&);
		
	private:
		Container						_container;
		nav_type 						_ddd;	
		DDTranslation 					_trans;
		DDRotationMatrix				_rot;
		DDSolidShape					_shape;
		DDName 							_ddname;
		GeometricEnumType 				_type;
		std::vector<double>				_params;
		DetId 							_geographicalID;
		std::vector< DDExpandedNode >	_parents;
		double 							_volume;
		double 							_density;
		double 							_weight;
		int    							_copy;
		std::string 					_material;
};

#endif
