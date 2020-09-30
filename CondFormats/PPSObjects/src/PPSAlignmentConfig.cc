/****************************************************************************
 *
 *  CondFormats/PPSObjects/interface/PPSAlignmentConfig.cc
 *
 *  Description : Class with alignment parameters
 *
 *  Authors:
 *  - Jan Kašpar
 *  - Mateusz Kocot
 *
 ****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"

#include "CondFormats/PPSObjects/interface/PPSAlignmentConfig.h"
TYPELOOKUP_DATA_REG(PPSAlignmentConfig);

#include <iostream>
#include <cmath>
#include <iomanip>

// -------------------------------- PPSAlignmentConfig getters --------------------------------

std::vector<std::string> PPSAlignmentConfig::sequence() const { return sequence_; }
std::string PPSAlignmentConfig::resultsDir() const { return resultsDir_; }

SectorConfig PPSAlignmentConfig::sectorConfig45() const { return sectorConfig45_; }
SectorConfig PPSAlignmentConfig::sectorConfig56() const { return sectorConfig56_; }

double PPSAlignmentConfig::x_ali_sh_step() const { return x_ali_sh_step_; }

double PPSAlignmentConfig::y_mode_sys_unc() const { return y_mode_sys_unc_; }
double PPSAlignmentConfig::chiSqThreshold() const { return chiSqThreshold_; }
double PPSAlignmentConfig::y_mode_unc_max_valid() const { return y_mode_unc_max_valid_; }
double PPSAlignmentConfig::y_mode_max_valid() const { return y_mode_max_valid_; }

double PPSAlignmentConfig::maxRPTracksSize() const { return maxRPTracksSize_; }
double PPSAlignmentConfig::n_si() const { return n_si_; }

std::map<unsigned int, std::vector<PointErrors>> PPSAlignmentConfig::matchingReferencePoints() const 
{ 
	return matchingReferencePoints_; 
}
std::map<unsigned int, SelectionRange> PPSAlignmentConfig::matchingShiftRanges() const 
{ 
	return matchingShiftRanges_; 
}

std::map<unsigned int, SelectionRange> PPSAlignmentConfig::alignment_x_meth_o_ranges() const 
{ 
	return alignment_x_meth_o_ranges_; 
}
unsigned int PPSAlignmentConfig::fitProfileMinBinEntries() const { return fitProfileMinBinEntries_; }
unsigned int PPSAlignmentConfig::fitProfileMinNReasonable() const { return fitProfileMinNReasonable_; }
unsigned int PPSAlignmentConfig::methOGraphMinN() const { return methOGraphMinN_; }
double PPSAlignmentConfig::methOUncFitRange() const { return methOUncFitRange_; }

std::map<unsigned int, SelectionRange> PPSAlignmentConfig::alignment_x_relative_ranges() const 
{ 
	return alignment_x_relative_ranges_; 
}
unsigned int PPSAlignmentConfig::nearFarMinEntries() const { return nearFarMinEntries_; }

std::map<unsigned int, SelectionRange> PPSAlignmentConfig::alignment_y_ranges() const 
{ 
	return alignment_y_ranges_; 
}
unsigned int PPSAlignmentConfig::modeGraphMinN() const { return modeGraphMinN_; }
unsigned int PPSAlignmentConfig::multSelProjYMinEntries() const { return multSelProjYMinEntries_; }

Binning PPSAlignmentConfig::binning() const { return binning_; }

// -------------------------------- PPSAlignmentConfig setters --------------------------------

void PPSAlignmentConfig::setSequence(std::vector<std::string> &sequence) { sequence_ = sequence; }
void PPSAlignmentConfig::setResultsDir(std::string &resultsDir) { resultsDir_ = resultsDir; }

void PPSAlignmentConfig::setSectorConfig45(SectorConfig &sectorConfig45) { sectorConfig45_ = sectorConfig45; }
void PPSAlignmentConfig::setSectorConfig56(SectorConfig &sectorConfig56) { sectorConfig56_ = sectorConfig56; }

void PPSAlignmentConfig::setX_ali_sh_step(double x_ali_sh_step) { x_ali_sh_step_ = x_ali_sh_step; }

void PPSAlignmentConfig::setY_mode_sys_unc(double y_mode_sys_unc) { y_mode_sys_unc_ = y_mode_sys_unc; }
void PPSAlignmentConfig::setChiSqThreshold(double chiSqThreshold) { chiSqThreshold_ = chiSqThreshold; }
void PPSAlignmentConfig::setY_mode_unc_max_valid(double y_mode_unc_max_valid) 
{ 
	y_mode_unc_max_valid_ = y_mode_unc_max_valid; 
}
void PPSAlignmentConfig::setY_mode_max_valid(double y_mode_max_valid) { y_mode_max_valid_ = y_mode_max_valid; }

void PPSAlignmentConfig::setMaxRPTracksSize(unsigned int maxRPTracksSize) { maxRPTracksSize_ = maxRPTracksSize; }
void PPSAlignmentConfig::setN_si(double n_si) { n_si_ = n_si; }

void PPSAlignmentConfig::setMatchingReferencePoints(std::map<unsigned int, std::vector<PointErrors>> &matchingReferencePoints)
{
	matchingReferencePoints_ = matchingReferencePoints;
}
void PPSAlignmentConfig::setMatchingShiftRanges(std::map<unsigned int, SelectionRange> &matchingShiftRanges)
{
	matchingShiftRanges_ = matchingShiftRanges;
}

void PPSAlignmentConfig::setAlignment_x_meth_o_ranges(std::map<unsigned int, SelectionRange> &alignment_x_meth_o_ranges)
{
	alignment_x_meth_o_ranges_ = alignment_x_meth_o_ranges;
}
void PPSAlignmentConfig::setFitProfileMinBinEntries(unsigned int fitProfileMinBinEntries)
{
	fitProfileMinBinEntries_ = fitProfileMinBinEntries;
}
void PPSAlignmentConfig::setFitProfileMinNReasonable(unsigned int fitProfileMinNReasonable)
{
	fitProfileMinNReasonable_ = fitProfileMinNReasonable;
}
void PPSAlignmentConfig::setMethOGraphMinN(unsigned int methOGraphMinN)
{
	methOGraphMinN_ = methOGraphMinN;
}
void PPSAlignmentConfig::setMethOUncFitRange(double methOUncFitRange)
{
	methOUncFitRange_ = methOUncFitRange;
}

void PPSAlignmentConfig::setAlignment_x_relative_ranges(std::map<unsigned int, SelectionRange> &alignment_x_relative_ranges)
{
	alignment_x_relative_ranges_ = alignment_x_relative_ranges;   
}
void PPSAlignmentConfig::setNearFarMinEntries(unsigned int nearFarMinEntries)
{
	nearFarMinEntries_ = nearFarMinEntries;
}

void PPSAlignmentConfig::setAlignment_y_ranges(std::map<unsigned int, SelectionRange> &alignment_y_ranges)
{
	alignment_y_ranges_ = alignment_y_ranges;
}
void PPSAlignmentConfig::setModeGraphMinN(unsigned int modeGraphMinN)
{
	modeGraphMinN_ = modeGraphMinN;
}
void PPSAlignmentConfig::setMultSelProjYMinEntries(unsigned int multSelProjYMinEntries)
{
	multSelProjYMinEntries_ = multSelProjYMinEntries;
}

void PPSAlignmentConfig::setBinning(Binning &binning) { binning_ = binning; }

// -------------------------------- << operators --------------------------------

std::ostream &operator<<(std::ostream &os, RPConfig &rc)
{
	os << std::fixed << std::setprecision(3);
	os << "    " << rc.name << ", id = " << rc.id << ", position = " << rc.position << ":\n";
	os << "        slope = " << rc.slope << ", sh_x = " << rc.sh_x << "\n";
	os << "        x_min_fit_mode = " << rc.x_min_fit_mode << ", x_max_fit_mode = " << rc.x_max_fit_mode << "\n";
	os << "        y_max_fit_mode = " << rc.y_max_fit_mode << "\n";
	os << "        y_cen_add = " << rc.y_cen_add << ", y_width_mult = " << rc.y_width_mult << "\n";
	os << std::setprecision(2);
	os << "        x slices: min = " << rc.x_slice_min << ", w = " << rc.x_slice_w << ", n = " 
	   << rc.x_slice_n;

	return os;
}

std::ostream &operator<<(std::ostream &os, SectorConfig &sc)
{
	// to be adjusted
	os << std::fixed << std::setprecision(3);
	os << sc.name << ":\n";
	os << sc.rp_N << "\n" << sc.rp_F << "\n";
	os << std::setprecision(3);
	os << "    slope = " << sc.slope << "\n";
	os << "    cut_h: apply = " << sc.cut_h_apply << ", a = " << sc.cut_h_a << ", c = " 
	   << sc.cut_h_c << ", si = " << sc.cut_h_si << "\n";
	os << "    cut_v: apply = " << sc.cut_v_apply << ", a = " << sc.cut_v_a << ", c = " 
	   << sc.cut_v_c << ", si = " << sc.cut_v_si << "\n";

	return os;
}

std::ostream &operator<<(std::ostream &os, Binning &b)
{
	os << "    bin_size_x = " << b.bin_size_x << ", n_bins_x = " << b.n_bins_x << "\n";
	os << "    pixel_x_offset = " << b.pixel_x_offset << "\n";
	os << "    n_bins_y = " << b.n_bins_y << ", y_min = " << b.y_min << ", y_max = " << b.y_max;

	return os;
}

std::ostream &operator<<(std::ostream &os, PPSAlignmentConfig c)
{
	os << "* sequence\n";
	for (unsigned int i = 0; i < c.sequence_.size(); i++)
	{
		os << "    " << i + 1 << ": " << c.sequence_[i] << "\n";
	}
	os << "\n";

	if (c.resultsDir_.empty())
	{
		os << "* no results file\n\n";
	}
	else 
	{
		os << "* results file directory:\n";
		os << "    " << c.resultsDir_ << "\n\n";
	}

	os << "* " << c.sectorConfig45_ << "\n\n";
	os << "* " << c.sectorConfig56_ << "\n\n";

	std::map<unsigned int, std::string> rpTags = {
		{ c.sectorConfig45_.rp_F.id, c.sectorConfig45_.rp_F.name },
		{ c.sectorConfig45_.rp_N.id, c.sectorConfig45_.rp_N.name },
		{ c.sectorConfig56_.rp_N.id, c.sectorConfig56_.rp_N.name },
		{ c.sectorConfig56_.rp_F.id, c.sectorConfig56_.rp_F.name }
	};

	os << "* x alignment shift step\n";
	os << "    x_ali_sh_step = " << c.x_ali_sh_step_ << "\n\n";

	os << "* mode graph parameters\n";
	os << "    y_mode_sys_unc = " << c.y_mode_sys_unc_ << "\n";
	os << "    chiSqThreshold = " << c.chiSqThreshold_ << "\n";
	os << "    y_mode_unc_max_valid = " << c.y_mode_unc_max_valid_ << "\n";
	os << "    y_mode_max_valid = " << c.y_mode_max_valid_ << "\n\n";

	os << "* selection\n";
	os << "    max_RP_tracks_size = " << c.maxRPTracksSize_ << "\n\n";

	os << "* cuts\n";
	os << "    n_si = " << c.n_si_ << "\n\n";

	os << "* matching\n" << std::setprecision(3);

	os << "    shift ranges:\n";
	for (const auto &p : c.matchingShiftRanges_)
		os << "        RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min 
		   << ", sh_max = " << p.second.x_max << "\n";

	os << "    reference points:\n";
	for (const auto &pm : c.matchingReferencePoints_)
	{
		os << "        " << std::setw(3) << pm.first << ": ";
		for (unsigned int i = 0; i < pm.second.size(); i++)
		{
			const auto &p = pm.second[i];
			if (i % 5 == 0 && i > 0)
				os << "\n             ";
			os << "(" << std::setw(6) << p.x << " +- " << p.ex << ", " << std::setw(6) << p.y << " +- " << p.ey << "), ";
		}
		os << "\n";
	}

	os << "\n" << "* alignment_x_meth_o\n";
	for (const auto &p : c.alignment_x_meth_o_ranges_)
		os << "    RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min 
		   << ", sh_max = " << p.second.x_max << "\n";
	os << "    fit_profile_min_bin_entries = " << c.fitProfileMinBinEntries_ << "\n";
	os << "    fit_profile_min_N_reasonable = " << c.fitProfileMinNReasonable_ << "\n";
	os << "    meth_o_graph_min_N = " << c.methOGraphMinN_ << "\n";
	os << "    meth_o_unc_fit_range = " << c.methOUncFitRange_ << "\n";

	os << "\n" << "* alignment_x_relative\n";
	for (const auto &p : c.alignment_x_relative_ranges_)
		os << "    RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min 
		   << ", sh_max = " << p.second.x_max << "\n";
	os << "    near_far_min_entries = " << c.nearFarMinEntries_ << "\n";

	os << "\n" << "* alignment_y\n";
	for (const auto &p : c.alignment_y_ranges_)
		os << "    RP " << rpTags[p.first] << " (" << std::setw(3) << p.first << "): sh_min = " << p.second.x_min 
		   << ", sh_max = " << p.second.x_max << "\n";
	os << "    mode_graph_min_N = " << c.modeGraphMinN_ << "\n";
	os << "    mult_sel_proj_y_min_entries = " << c.multSelProjYMinEntries_ << "\n";

	os << "\n" << "* binning\n";
	os << c.binning_ << "\n";

	return os;
}