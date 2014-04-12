#ifndef DDCore_DDCategory_h
#define DDCore_DDCategory_h

//! enumaration of a possible categorization of the DDLogicalPart, defaults to unspecified
// FIXME: use namespaces as soon as there's a clear CMS strategy for them
struct DDEnums {
  enum Category { unspecified, sensitive, cable, cooling, support, envelope };
  enum Shapes { not_init, box, tubs, trap, cons, 
                polycone_rz, polycone_rrz,
		polyhedra_rz, polyhedra_rrz,
	        b_union, b_subtraction, b_intersection,
		reflected,
		shapeless,
		pseudotrap
	      };

  static const char * const categoryName(Category s)
  {
    static const char * const c[] = { "unspecified", "sensitive", "cable", "cooling", "support", "envelope" };
    return c[s];
  }	       	      
};


#endif
