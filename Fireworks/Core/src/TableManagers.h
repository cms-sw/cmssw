// -*- C++ -*-
#ifndef Fireworks_Core_TableManagers_h
#define Fireworks_Core_TableManagers_h

#include "TableWidget.h"
#include "LightTableWidget.h"
#include <vector>
#include <stdio.h>

class FWEventItem;

// go from row in the table to index in the underlying collection
template <class T> int table_row_to_index (const std::vector<T> &v, int row)
{
     if ((unsigned int)row < v.size())
	  return v[row].index;
     else return -1;
}

// go from index in the underlying collection to row in the table
template <class T> int index_to_table_row (const std::vector<T> &v, int idx)
{
     for (unsigned int i = 0; i < v.size(); ++i)
	  if (v[i].index == idx)
	       return i;
     return 0;
}

class FWTableManager : public LightTableManager {
public:
     FWTableManager ();
     // can do all the things a TableManager can, but is also
     // text-dumpable
     virtual void dump (FILE *);
     //void sort (int col, bool reset = false);
     // and has a utility for making a display frame
     void MakeFrame (TGCompositeFrame *parent, int width, int height,
		     unsigned int layout);
     void Update (int rows = 5);
     void Selection (int row, int mask);
     void selectRows ();
     virtual int table_row_to_index (int) const { return 0; }
     virtual int index_to_table_row (int) const { return 0; }
     virtual bool rowIsSelected(int row) const {
        return sel_indices.count(table_row_to_index(row));
     }
     virtual bool rowIsVisible (int row) const {
        return vis_indices.count(table_row_to_index(row));
     }
     virtual bool idxIsSelected (int idx) const {
	  return sel_indices.count(idx);
     }
     virtual bool idxIsVisible (int idx) const {
	  return vis_indices.count(idx);
     }
     void setItem (FWEventItem *);
     void itemGoingToBeDestroyed ();

public:
     LightTableWidget 	*widget;
     TGCompositeFrame	*frame;
     TGTextEntry	*title_frame;
     FWEventItem	*item;
     std::set<int> 	sel_indices;
     std::set<int> 	vis_indices;
     //int		sort_col_;
     //bool		sort_asc_;
};

std::string format_string (const std::string &fmt, int x);
std::string format_string (const std::string &fmt, double x);

template <class Row> struct sort_asc {
     sort_asc (FWTableManager *m) : manager(m) { }
     int i;
     bool order;
     FWTableManager *manager;
     bool operator () (const Row &r1, const Row &r2) const
	  {
	       // visible rows always win over invisible rows
	       bool r1_vis = manager->idxIsVisible(r1.index);
	       bool r2_vis = manager->idxIsVisible(r2.index);
	       if (r1_vis && !r2_vis)
		    return true;
	       if (!r1_vis && r2_vis)
		    return false;
	       // otherwise the column content decides
	       if (order)
		    return r1.vec()[i] > r2.vec()[i];
	       else return r1.vec()[i] < r2.vec()[i];
	  }
};

enum { FLAG_NO, FLAG_YES, FLAG_MAYBE };

struct ElectronRowStruct {
     int	index;
     float 	Et, eta, phi, eop, hoe, fbrem, dei, dpi;
//      float	see, spp, iso;
//      char 	robust, loose, tight;
};

class ElectronRow : public ElectronRowStruct {
public:
     ElectronRow (const ElectronRowStruct &e) : ElectronRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

#if 0
// you pay for this:
template <class Row> class ObjectTableManager : public FWTableManager {
public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "Electrons"; }

     std::vector<Row>		rows;
     static std::string		titles[];
     static std::string		formats[];
};
typedef ObjectTableManager<ElectronRow> ElectronTableManager;
#else
// and they give you that:
class ElectronTableManager : public FWTableManager {
public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "Electrons"; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<ElectronRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};
#endif

struct MuonRowStruct {
     int	index;
     float 	pt;
     char 	global, tk, SA, calo;
//      float 	iso_3, iso_5;
     float	tr_pt, eta, phi;
//      float	chi2_ndof, matches;
     float	d0, sig_d0;
//      char	loose_match, tight_match, loose_depth, tight_depth;
};

class MuonRow : public MuonRowStruct {
public:
     MuonRow (const MuonRowStruct &e) : MuonRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

class MuonTableManager : public FWTableManager {

public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "Muons"; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<MuonRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct JetRowStruct {
     int	index;
     float	Et, eta, phi, ECAL, HCAL, emf;
//      float	chf;
};

class JetRow : public JetRowStruct {
public:
     JetRow (const JetRowStruct &e) : JetRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

class JetTableManager : public FWTableManager {

public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "Jets"; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<JetRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct TrackRowStruct {
     int	index;
     float	pt,
	  eta,
	  phi,
	  d0,
	  d0_err,
	  z0,
	  z0_err,
	  vtx_x,
	  vtx_y,
	  vtx_z;
     int 	pix_layers, strip_layers;
//      int	outermost_layer;
     float chi2;
     float ndof;
};

class TrackRow : public TrackRowStruct {
public:
     TrackRow (const TrackRowStruct &e) : TrackRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

class TrackTableManager : public FWTableManager {

public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "Tracks"; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<TrackRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct VertexRowStruct {
     int	index;
     float 	vx,
	  vx_err,
	  vy,
	  vy_err,
	  vz,
	  vz_err;
     int	n_tracks;
     float	chi2;
     float	ndof;
};

class VertexRow : public VertexRowStruct {
public:
     VertexRow (const VertexRowStruct &e) : VertexRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

class VertexTableManager : public FWTableManager {

public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "Vertices"; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<VertexRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct HLTRowStruct {
     int	index;
     float	pt,
	  eta,
	  phi,
	  d0,
	  d0_err,
	  z0,
	  z0_err,
	  vtx_x,
	  vtx_y,
	  vtx_z;
     int 	pix_layers,
	  strip_layers,
	  outermost_layer;
     float chi2;
     float ndof;
};

class HLTRow : public HLTRowStruct {
public:
     HLTRow (const HLTRowStruct &e) : HLTRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

class HLTTableManager : public FWTableManager {

public:
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return "HLT names"; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<HLTRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct L1RowStruct {
     int	index;
     float	Et, eta, phi;
     char	type[100];
};

class L1Row : public L1RowStruct {
public:
     L1Row (const L1RowStruct &e) : L1RowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

class L1TableManager : public FWTableManager {

public:
     L1TableManager () : title_("L1 objects") { }
     virtual int NumberOfRows() const;
     virtual int NumberOfCols() const;
     virtual void Sort(int col, bool sortOrder); // sortOrder=true means desc order
     virtual std::vector<std::string> GetTitles(int col);
     virtual void FillCells(int rowStart, int colStart,
			    int rowEnd, int colEnd,
			    std::vector<std::string>& oToFill);
     virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame);
     virtual void UpdateRowCell(int row, TGFrame *rowCell);
     const std::string		title () const { return title_; }
     void	setTitle (const std::string &t) { title_ = t; }

     virtual int table_row_to_index (int i) const
	  { return ::table_row_to_index(rows, i); }
     virtual int index_to_table_row (int i) const
	  { return ::index_to_table_row(rows, i); }

     std::vector<L1Row>		rows;
     static std::string		titles[];
     static std::string		formats[];
     std::string		title_;
};

#endif
