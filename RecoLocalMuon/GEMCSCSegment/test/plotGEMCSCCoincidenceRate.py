#!/usr/bin/env python3.9
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Optional
import argparse
from pathlib import Path
import numpy as np
import uproot
import hist
from hist import intervals
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

__author__ = "Seungjin Yang"
__email__ = "seungjin.yang@cern.ch"
__version__ = "CMSSW_12_6_0_pre2"

@dataclass
class Efficiency1D:
    x: np.ndarray
    y: np.ndarray
    ylow: np.ndarray
    yup: np.ndarray

    @classmethod
    def from_hist(cls, h_num, h_den) -> Efficiency1D:
        num = h_num.values()
        den = h_den.values()

        x = h_den.axes[0].centers
        y = np.divide(num, den, where=den > 0)
        ylow, yup = intervals.clopper_pearson_interval(num, den)
        return cls(x, y, ylow, yup)

    @cached_property
    def yerr_low(self) -> np.ndarray:
        return self.y - self.ylow

    @cached_property
    def yerr_up(self) -> np.ndarray:
        return self.yup - self.y

    @property
    def yerr(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.yerr_low, self.yerr_up)

    def plot(self, ax: Optional[mpl.axes.Subplot] = None, **kwargs):
        ax = ax or plt.gca()
        return ax.errorbar(self.x, self.y, self.yerr, **kwargs)

def plot_eff(eff_twofold_all: Efficiency1D,
        eff_twofold_muon: Efficiency1D,
        eff_threefold_all: Efficiency1D,
        eff_threefold_muon: Efficiency1D,
        gem_lable: str,
        output_dir: Path):
    fig, ax = plt.subplots(figsize=(16, 8))

    eff_twofold_all.plot(ax=ax, label='Twofold', ls='', marker='s')
    eff_twofold_muon.plot(ax=ax, label='Twofold & Muon', ls='', marker='s')
    eff_threefold_all.plot(ax=ax, label='Threefold', ls='', marker='o')
    eff_threefold_muon.plot(ax=ax, label='Threefold & Muon', ls='', marker='o')

    ax.set_xlabel('Chamber')
    ax.set_ylabel('Efficiency')

    ax.set_xticks(eff_twofold_all.x)
    ax.grid()
    ax.set_ylim(0.5, 1)
    ax.legend(title=gem_lable, ncol=2, loc='lower center')

    fig.tight_layout()

    output_path = output_dir / gem_lable
    for suffix in ['.png', '.pdf']:
        fig.savefig(output_path.with_suffix(suffix))

def name_gem_layer(region: int, station: int, layer: int) -> str:
    if region == 1:
        region_char = 'P'
    elif region == -1:
        region_char = 'M'
    else:
        raise ValueError(f'{region=:}')
    return f'GE{station}1-{region_char}-L{layer}'


def plot_layer(tree, region: int, station: int, output_dir: Path):
    expressions = [
        'region',
        'station',
        'chamber',
        'csc_is_muon',
        'gem_layer'
    ]

    cut_list = [
        f'region == {region}',
        f'station == {station}',
        'ring == 1',
        'csc_num_hit == 6',
        '~gem_chamber_has_error',
    ]
    cut = ' & '.join(f'({each})' for each in cut_list)

    num_chambers = 36 if station == 1 else 18
    chamber_axis = hist.axis.Regular(num_chambers, 0.5, num_chambers + 0.5, name='chamber', label='Chamber')
    muon_axis = hist.axis.hist.axis.Boolean(name='muon')

    # has a csc segment
    h_csc = hist.Hist(chamber_axis, muon_axis, storage=hist.storage.Int64())
    # has a CSC segment and a hit on L1(L2) layer
    h_l1 = h_csc.copy()
    h_l2 = h_csc.copy()
    # has a csc segment and hits on both layers
    h_both = h_csc.copy()

    for chunk in tree.iterate(expressions, cut=cut):
        has_l1 = ak.any(chunk.gem_layer == 1, axis=1)
        has_l2 = ak.any(chunk.gem_layer == 2, axis=1)
        has_both = has_l1 & has_l2

        h_csc.fill(chamber=chunk.chamber, muon=chunk.csc_is_muon)
        h_l1.fill(chamber=chunk.chamber[has_l1], muon=chunk.csc_is_muon[has_l1])
        h_l2.fill(chamber=chunk.chamber[has_l2], muon=chunk.csc_is_muon[has_l2])
        h_both.fill(chamber=chunk.chamber[has_both], muon=chunk.csc_is_muon[has_both])

    # twofold efficiencies
    eff_l1_twofold_all = Efficiency1D.from_hist(h_num=h_l1.project(0), h_den=h_csc.project(0))
    eff_l2_twofold_all = Efficiency1D.from_hist(h_num=h_l2.project(0), h_den=h_csc.project(0))

    # threefold efficiencies
    eff_l1_threefold_all = Efficiency1D.from_hist(h_num=h_both.project(0), h_den=h_l2.project(0))
    eff_l2_threefold_all = Efficiency1D.from_hist(h_num=h_both.project(0), h_den=h_l1.project(0))

    # efficiencies
    muon_mask = (slice(None), True)
    eff_l1_twofold_muon = Efficiency1D.from_hist(h_num=h_l1[muon_mask], h_den=h_csc[muon_mask])
    eff_l2_twofold_muon = Efficiency1D.from_hist(h_num=h_l2[muon_mask], h_den=h_csc[muon_mask])
    eff_l1_threefold_muon = Efficiency1D.from_hist(h_num=h_both[muon_mask], h_den=h_l2[muon_mask])
    eff_l2_threefold_muon = Efficiency1D.from_hist(h_num=h_both[muon_mask], h_den=h_l1[muon_mask])

    plot_eff(eff_twofold_all=eff_l1_twofold_all,
             eff_twofold_muon=eff_l1_twofold_muon,
             eff_threefold_all=eff_l1_threefold_all,
             eff_threefold_muon=eff_l1_threefold_muon,
             gem_lable=name_gem_layer(region, station, layer=1),
             output_dir=output_dir)

    plot_eff(eff_twofold_all=eff_l2_twofold_all,
             eff_twofold_muon=eff_l2_twofold_muon,
             eff_threefold_all=eff_l2_threefold_all,
             eff_threefold_muon=eff_l2_threefold_muon,
             gem_lable=name_gem_layer(region, station, layer=2),
             output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-path', type=Path, required=True, help='input file')
    parser.add_argument('-t', '--treepath', type=str, default='gemcscCoincidenceRateAnalyzer/gemcsc', help='tree path')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path.cwd(), help='output directory')
    parser.add_argument('--ge11', action=argparse.BooleanOptionalAction, default=True, help='plot GE11')
    parser.add_argument('--ge21', action=argparse.BooleanOptionalAction, default=True, help='plot GE21')
    args = parser.parse_args()

    if not args.ge11 and not args.ge21:
        raise RuntimeError

    tree = uproot.open(f'{args.input_path}:{args.treepath}')

    region_list = [-1, 1]
    station_list = []
    if args.ge11:
        station_list.append(1)
    if args.ge21:
        station_list.append(2)

    if not args.output_dir.exists():
        print(f'mkdir -p {args.output_dir}')
        args.output_dir.mkdir(parents=True)

    for region in region_list:
        for station in station_list:
            print(f'plotting layers in {region=} & {station=}')
            plot_layer(tree=tree, region=region, station=station, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
