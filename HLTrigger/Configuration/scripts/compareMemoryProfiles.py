#!/usr/bin/env python3

"""
CMS-style hardware memory comparison plotter.

Reads one or more CSV memory-profile files (time series of per-GPU or
total memory usage in MiB) and produces a comparison plot in CMS style.

Features
--------
- CMS plotting style (mplhep)
- Automatic MiB -> GiB conversion
- Peak memory annotations
- Optional hardware-limit reference line
- Optional informational text box
- PDF + PNG output

Example
-------
python3 compareMemoryProfiles.py \
    gpu_memory_NewBaseline.csv gpu_memory_Current.csv \
    --csv-labels NewBaseline Current \
    --cms-label "PR Testing" \
    --outfile memory_comparison
"""

import argparse
from itertools import cycle
from pathlib import Path

# Headless-safe backend for CI environments
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

DEFAULT_COLOURS = [
    '#3f8fda', '#bd1f01', '#94a4a2', '#832db6', '#a96b59',
    '#e76300', '#b9ac70', '#717581', '#92dadd', '#ffa90e',
]

def find_plateau_segments(time, mem, plateau_fraction, min_plateau_points):
    """
    Find contiguous segments where memory usage is at or above
    `plateau_fraction` of the maximum value, with at least
    `min_plateau_points` samples.
    """
    threshold = plateau_fraction * mem.max()
    mask = mem >= threshold

    segments = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_plateau_points:
                segments.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_plateau_points:
        segments.append((start, len(mask) - 1))

    return segments

def plot_gpu_memory(
    csv_files: list[str | Path],
    legend_entries: list[str],
    colours: list[str] | None = None,
    # per_gpu=True: plot one curve per GPU (columns 1..-2); False: plot total (last column)
    per_gpu: bool = False,
    # Plateau detection (set plateau_fraction=None to disable)
    plateau_fraction: float | None = 0.75,
    min_plateau_points: int = 30,
    streams_per_gpu: int = 256,
    # Figure layout
    figsize: tuple[float, float] = (10, 8),
    # X axis
    xlims: tuple[float, float] = (0, 2500),
    # Y axis
    ylims: tuple[float, float] = (0, 60),
    major_tick_step: float = 10,
    minor_tick_step: float = 2,
    # Axis labels
    xlabel: str = "Elapsed time [s]",
    ylabel: str = "Memory usage [GiB]",
    xlabel_fontsize: int = 22,
    ylabel_fontsize: int = 22,
    # CMS label
    cms_label: str = "Internal",
    cms_data: bool = False,
    cms_loc: int = 1,
    cms_rlabel: str = "",
    cms_fontsize: int = 20,
    # Info text box (set to None to omit)
    info_text: list[str] | None = None,
    info_xy: tuple[float, float] = (0.05, 0.9),
    info_fontsize: int = 15,
    # Hardware limit line (set to None to omit)
    hw_limit: float | None = None,
    hw_limit_label: str = "Hardware limit",
    # Legend
    legend_loc: str = "upper right",
    legend_fontsize: int = 16,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more GPU memory-usage time series in CMS style.

    Parameters
    ----------
    csv_files : list[Path or str]
        Input CSV files. Each file is expected to have no header row
        actually present in the data (the first row is skipped), with
        column 0 = elapsed time [s], columns 1..-2 = per-GPU memory
        [MiB], and column -1 = total memory [MiB].
    legend_entries : list[str]
        Legend label corresponding to each input file.
    colours : list[str], optional
        Colour cycle to use. Defaults to DEFAULT_COLOURS.
    per_gpu : bool
        If True, plot one curve per GPU column. If False, plot only
        the total (last column).
    plateau_fraction : float or None
        If set, detect "plateau" segments where memory usage is above
        this fraction of the curve's maximum, and report their mean.
        Set to None to disable plateau detection entirely.
    min_plateau_points : int
        Minimum number of samples for a plateau segment to be reported.
    streams_per_gpu : int
        Used to convert the plateau mean memory into an estimated
        per-stream memory footprint (printed only).
    figsize : tuple
        Figure size in inches.
    xlims : tuple
        (min, max) y-axis limits (memory, in GiB).
    ylims : tuple
        (min, max) y-axis limits (memory, in GiB).
    major_tick_step, minor_tick_step : float
        Spacing of major/minor y-axis ticks.
    xlabel, ylabel : str
        Axis labels.
    xlabel_fontsize, ylabel_fontsize : int
        Axis label font sizes.
    cms_label : str
        Text passed to mplhep's CMS label (e.g. "Preliminary").
    cms_loc : int
        Passed through to hep.cms.label's `loc` argument.
    cms_rlabel : str
        Right-hand label text for hep.cms.label.
    cms_fontsize : int
        Font size for the CMS label.
    info_text : str or None
        Optional multi-line text drawn in axes coordinates (e.g.
        dataset/hardware description). Set to None to omit.
    info_xy : tuple
        (x, y) position of info_text in axes coordinates.
    info_fontsize : int
        Font size for info_text.
    hw_limit : float or None
        If set, draw a horizontal red line at this y-value to mark a
        hardware memory limit.
    hw_limit_label : str
        Label drawn next to the hardware-limit line.
    legend_loc : str
        Matplotlib legend location string.
    legend_fontsize : int
        Legend font size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects.
    """
    colours = colours or DEFAULT_COLOURS
    colour_iter = cycle(colours)
    fig, ax = plt.subplots(figsize=figsize)

    max_time, min_time = -1., 1E9
    
    for file, legend_entry in zip(csv_files, legend_entries):
        raw = pd.read_csv(file, header=None, skiprows=1)
        time = raw.iloc[:, 0].to_numpy()
        if time[0] < min_time:
            min_time = time[0]
        if time[-1] > max_time:
            max_time = time[-1]

        if per_gpu:
            n_gpus = raw.shape[1] - 2  # exclude time col and total col
            curves = [
                (raw.iloc[:, i + 1].to_numpy() / 1024, f"{legend_entry} – GPU {i}")
                for i in range(n_gpus)
            ]
        else:
            curves = [(raw.iloc[:, -1].to_numpy() / 1024, legend_entry)]

        overall_max_mem = 0.
        for mem, label in curves:
            colour = next(colour_iter)
            ax.plot(time, mem, label=label, color=colour, linewidth=2)

            print(f"Label: {label}")
            print(f"  Max Memory: {mem.max():.1f} GiB")

            if plateau_fraction is not None:
                segments = find_plateau_segments(
                    time, mem, plateau_fraction, min_plateau_points
                )
                if not segments:
                    print(
                        f"  -> No plateau segments detected (adjust parameters)."
                    )
                else:
                    total_points = 0
                    weighted_sum = 0.0
                    for i, (s, e) in enumerate(segments, start=1):
                        seg_mean = mem[s : e + 1].mean()
                        npts = e - s + 1
                        total_points += npts
                        weighted_sum += seg_mean * npts
                        print(
                            f"  Plateau {i}: {time[s]:.1f}s -> {time[e]:.1f}s  "
                            f"(n={npts})  mean={seg_mean:.1f} GiB"
                        )

                    weighted_mean = (
                        weighted_sum / total_points if total_points else float("nan")
                    )
                    print(f"  Weighted plateau mean: {weighted_mean:.1f} GiB")
                    print(
                        "  Average memory usage per stream per GPU (plateau): "
                        f"{weighted_mean * 1024 / streams_per_gpu:.1f} MiB"
                    )

            print("---")

            max_mem = mem.max()
            ax.hlines(max_mem, xlims[0], xlims[1], colors=colour, linestyles=":", linewidth=1.4)
            ax.text(xlims[1] * 0.98, max_mem, f"{max_mem:.1f} GiB",
                    verticalalignment="center", horizontalalignment="right",
                    color=colour, fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=colour, alpha=0.8))

            overall_max_mem = max(overall_max_mem, max_mem)
            
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    label_obj = hep.cms.label(cms_label, ax=ax, loc=cms_loc, rlabel=cms_rlabel, fontsize=cms_fontsize)
    label_obj[0]._y += 0.02
    label_obj[1]._y += 0.02

    if info_text is not None:
        x0, y0 = info_xy
        y_step = 0.03
        x_step = 0.001
        for i, line in enumerate(info_text):
            # not clear why x_step is needed for perfect alignment
            ax.text(x0 - i * x_step, y0 - i * y_step, line, transform=ax.transAxes,
                    fontsize=info_fontsize, va="top", ha="left", style="italic")

    if hw_limit is not None:
        ax.hlines(hw_limit, xlims[0], xlims[1], colors="red", linestyles="-", linewidth=1.4)
        ax.text(xlims[1] * 0.98, hw_limit, hw_limit_label,
                verticalalignment="center", horizontalalignment="right",
                color="red", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=1))

    ax.legend(loc=legend_loc, frameon=False, fontsize=legend_fontsize)

    if xlims == (0, 0):
        ax.set_xlim(min_time, max_time + 0.2*(max_time - min_time))
    else:
        ax.set_xlim(*xlims)
    if ylims == (0, 0):
        true_ymin, true_ymax = ax.set_ylim(0., 1.4*overall_max_mem)
    else:
        true_ymin, true_ymax = ax.set_ylim(*ylims)

    major_ticks = np.arange(true_ymin, true_ymax + 1, major_tick_step)
    minor_ticks = np.arange(true_ymin, true_ymax + 1, minor_tick_step)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    ax.grid(True, which="major", alpha=0.8)
    ax.grid(True, which="minor", alpha=0.4, linestyle=":")

    plt.tight_layout()
    return fig, ax


def main(args):
    fig, ax = plot_gpu_memory(
        csv_files=args.csv_files,
        legend_entries=args.csv_labels if len(args.csv_labels) > 0 else [str(x) for x in range(len(args.csv_files))],
        colours=args.colours,
        plateau_fraction=None,
        per_gpu=False,
        figsize=(12, 9),
        xlims=args.xlims,
        ylims=args.ylims,
        major_tick_step=args.major_tick_step,
        minor_tick_step=2,
        cms_label=args.cms_label,
        cms_loc=1,
        cms_rlabel='',
        cms_fontsize=18,
        info_text=args.info_text,
        info_xy=(0.05, 0.92),
        hw_limit=None,
        legend_loc='upper right',
    )

    if args.title:
        ax.set_title(args.title)

    outfile_base = args.outfile
    outfile_base.parent.mkdir(parents=True, exist_ok=True)

    for ext in ('png', 'pdf'):
        figname = outfile_base.with_name(f"{outfile_base.name}.{ext}")
        fig.savefig(figname, dpi=300, bbox_inches="tight")
        print(f"Saved {figname}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CMS-style hardware memory comparison plotter.'
    )

    parser.add_argument(
        'csv_files',
        nargs='+',
        type=Path,
        help='Input CSV files (1 or more).'
    )

    parser.add_argument(
        '--csv-labels',
        nargs='+',
        type=str,
        required=False,
        help='Legend labels, one per input CSV file (same order, same count).'
    )

    parser.add_argument(
        '--colours',
        nargs='+',
        type=str,
        required=False,
        default=DEFAULT_COLOURS,
        help='Colours in hex format, one per input CSV file (same order, same count).'
    )

    parser.add_argument(
        '--xlims',
        type=float,
        nargs=2,
        required=False,
        default=(0., 0.),
        help='X axis limits [xmin, xmax]. If ignored, an automatic range is employed based on the overall maximum memory value.'
    )

    parser.add_argument(
        '--ylims',
        type=float,
        nargs=2,
        required=False,
        default=(0., 0.),
        help='Y axis limits [ymin, ymax]. If ignored, an automatic range is employed based on the overall maximum memory value.'
    )

    parser.add_argument(
        '--major-tick-step',
        type=float,
        required=False,
        default=10,
        help='The separation between two major ticks in Y axis units.'
    )

    parser.add_argument(
        '--title',
        help='Optional plot title.',
    )

    parser.add_argument(
        '--outfile',
        default=Path('memory_comparison'),
        type=Path,
        help='Output base filename (without extension).'
    )

    parser.add_argument(
        '--cms-label',
        default='CMSSW integration testing.',
        help='CMS label text',
    )

    info_text_default = (
        'tt + 200 PU (s = 14 TeV)',
        '2x AMD EPYC 9534 64-Core Processor',
        '4x NVIDIA L40S GPUs',
        '16 jobs with 16 threads/streams each',
    )
    parser.add_argument(
        '--info-text',
        nargs='+',
        default=info_text_default,
        help='Informative text to add below the CMS label in the top left corner.',
    )

    args = parser.parse_args()

    if len(args.csv_files) != len(args.csv_labels):
        parser.error('Number of --csv-labels must match number of csv_files.')

    main(args)
