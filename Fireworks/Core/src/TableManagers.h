// -*- C++ -*-
#ifndef Fireworks_Core_TableManagers_h
#define Fireworks_Core_TableManagers_h

#include "TableWidget.h"
#include <vector>
#include <stdio.h>

class FWEventItem;

// go from row in the table to index in the underlying collection
template <class T> int table_row_to_index (const std::vector<T> &v, int row)
{
     return v[row].index;
}

// go from index in the underlying collection to row in the table
template <class T> int index_to_table_row (const std::vector<T> &v, int idx)
{
     for (unsigned int i = 0; i < v.size(); ++i)
	  if (v[i].index == idx)
	       return i;
     return 0;
}

class FWTableManager : public TableManager {
public:
     FWTableManager ();
     // can do all the things a TableManager can, but is also
     // text-dumpable
     virtual void dump (FILE *);
     // and has a utility for making a display frame
     void MakeFrame (TGCompositeFrame *parent, int width, int height);
     void Update (int rows = 5);
     void Selection (int row, int mask);
     void selectRows ();
     virtual int table_row_to_index (int) const { return 0; }
     virtual int index_to_table_row (int) const { return 0; }

public:
     TableWidget 	*widget;
     TGCompositeFrame	*frame;
     TGTextEntry	*title_frame;
     FWEventItem	*item;
     std::set<int> 	sel_indices;
};

std::string format_string (const std::string &fmt, int x);
std::string format_string (const std::string &fmt, double x);

enum { FLAG_NO, FLAG_YES, FLAG_MAYBE };

struct ElectronRowStruct {
     int	index;
     float 	Et, eta, phi, eop, hoe, fbrem, dei, dpi, see, spp, iso;
     char 	robust, loose, tight;
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
     float 	iso_3, iso_5, tr_pt, eta, phi, chi2_ndof, matches, d0, sig_d0;
     char	loose_match, tight_match, loose_depth, tight_depth;
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
     float	Et, eta, phi, ECAL, HCAL, emf, chf;
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
     int 	pix_layers,
	  strip_layers,
	  outermost_layer;
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

#endif
