#ifndef FastSimulation_CaloGeometryTools_CrystalNeighbour_h
#define FastSimulation_CaloGeometryTools_CrystalNeighbour_h
/** 
 * 
 * Stores basic information on the neighbour. Used in EcalHitMaker
 *
 * \author Florian Beaudette
 * \date: 08-Jun-2004 - 06-Oct-2006
 */
class CrystalNeighbour {
public:
  CrystalNeighbour(unsigned number = 999, int status = -2) : number_(number), status_(status) { ; }
  ~CrystalNeighbour() { ; };
  /// Number of the crystal. This has nothing to do with the UniqueID
  inline unsigned number() const { return number_; };
  /// get the status 0 : gap; 1: crack ; -1 : does not exist ; -2 not calculated yet
  inline int status() const { return status_; };
  /// set the status
  inline void setStatus(int status) { status_ = status; };
  /// set the number
  inline void setNumber(unsigned n) { number_ = n; };

  /// set if this direction should be projected
  /// this means something only if the direction is N,E,W,S
  inline void setToBeGlued(bool proj) { tobeprojected_ = proj; };

  /// do the edge in this direction should be glued ?
  inline bool toBeGlued() const { return tobeprojected_; };

private:
  unsigned number_;
  int status_;
  bool tobeprojected_;
};
#endif
