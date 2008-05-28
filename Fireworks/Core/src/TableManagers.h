// -*- C++ -*-
#ifndef Fireworks_Core_TableManagers_h
#define Fireworks_Core_TableManagers_h

#include "TableWidget.h"

class FWTableManager : public TableManager {
public:
     // can do all the things a TableManager can, but is also
     // text-dumpable
     void Dump () { }
     // and has a utility for making a display frame
     void MakeFrame (TGMainFrame *parent, int width, int height);
     void Update () { widget->InitTableCells(); widget->UpdateTableCells(0, 0); }
public:
     TableWidget 	*widget;
     TGCompositeFrame	*frame;
     TGTextEntry	*title_frame;
};

std::string format_string (const std::string &fmt, int x);
std::string format_string (const std::string &fmt, double x);

enum { FLAG_NO, FLAG_YES, FLAG_MAYBE };

struct ElectronRowStruct {
     float 	Et, eta, phi, eop, hoe, fbrem, dei, dpi, see, spp, iso;
     char 	robust, loose, tight;
};

struct ElectronRow : public ElectronRowStruct {
public:
     ElectronRow (const ElectronRowStruct &e) : ElectronRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
     const std::vector<float> 		&vec () const;
protected:
     mutable std::vector<std::string>	str_;
     mutable std::vector<float>		vec_;
};

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

     std::vector<ElectronRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct MuonRowStruct {
     float 	pt;
     char 	global, tk, SA, calo;
     float 	iso_3, iso_5, tr_pt, eta, phi, chi2_ndof, matches, d0, sig_d0;
     char	loose_match, tight_match, loose_depth, tight_depth;
};

struct MuonRow : public MuonRowStruct {
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

     std::vector<MuonRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

struct JetRowStruct {
     float	Et, eta, phi, ECAL, HCAL, emf, chf;
};

struct JetRow : public JetRowStruct {
public:
     JetRow (const JetRowStruct &e) : JetRowStruct(e) { }
     const std::vector<std::string> 	&str () const;
protected:
     mutable std::vector<std::string>	str_;
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

     std::vector<JetRow>	rows;
     static std::string		titles[];
     static std::string		formats[];
};

#endif
