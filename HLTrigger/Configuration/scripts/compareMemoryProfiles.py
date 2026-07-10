#!/usr/bin/env python3

"""
CMS-style hardware memory comparison plotter.

Features
--------
- CMS plotting style (mplhep)
- Automatic MiB -> GiB conversion
- Default labels from filenames
- Peak memory annotations
- Relative peak difference calculation
- Average memory usage calculation
- Automatic PDF + PNG output
- Metadata box
- Optional interactive display
- Option to display CPU or GPU

Example
-------
python3 compareMemoryProfiles.py \
    gpu_memory_HLTTimingNewBaseline_16j_16t_16s.csv \
    gpu_memory_HLTTimingCurrent_16j_16t_16s.csv \
    --label1 NewBaseline \
    --label2 Current \
    --cms-label "PR Testing"

Typical CMSSW integration usage
-------------------------------
python3 compareMemoryProfiles.py \
    baseline.csv current.csv \
    --output gpu_memory_PR_comparison \
    --no-show
"""

import argparse
from pathlib import Path

# Headless-safe backend for CI environments
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import numpy as np


def read_file(path):
    """Read CSV input file."""
    return pd.read_csv(path)


def default_label(path):
    """Generate default legend label from filename stem."""
    return Path(path).stem


def compute_stats(df):
    """Compute useful memory statistics."""
    peak = df["memory_gib"].max()
    avg = df["memory_gib"].mean()

    peak_idx = df["memory_gib"].idxmax()

    peak_time = df.loc[peak_idx, "elapsed_seconds"]

    return {
        "peak": peak,
        "avg": avg,
        "peak_time": peak_time,
    }


def annotate_peak(ax, stats, color):
    """Annotate peak memory point."""
    ax.scatter(
        stats["peak_time"],
        stats["peak"],
        s=50,
        zorder=5,
        color=color,
    )

    ax.annotate(
        f'{stats["peak"]:.2f} GiB',
        xy=(stats["peak_time"], stats["peak"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        color=color,
    )


def save_outputs(fig, output_base, isGPU):
    """Save figure in multiple formats."""

    prefix = "gpu" if isGPU else "cpu"
    png_name = f"{prefix}_{output_base}.png"
    pdf_name = f"{prefix}_{output_base}.pdf"
    
    fig.savefig(
        png_name,
        dpi=220,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_name,
        bbox_inches="tight",
    )

    print(f"Saved: {png_name}")
    print(f"Saved: {pdf_name}")


def main(args):
    # CMS plotting style
    plt.style.use(hep.style.CMS)

    # Read data
    df1 = read_file(args.file1)

    # Convert MiB -> GiB
    df1["memory_gib"] = df1["memory_mib"] / 1024.0
        
    # Labels
    label1 = args.label1 or default_label(args.file1)

    # Statistics
    stats1 = compute_stats(df1)

    if args.file2:
        df2 = read_file(args.file2)
        df2["memory_gib"] = df2["memory_mib"] / 1024.0
        label2 = args.label2 or default_label(args.file2)
        stats2 = compute_stats(df2)
        rel_diff = 100.0 * (stats2["peak"] - stats1["peak"]) / stats1["peak"]

    # Figure
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Colors from CMS default cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot curves
    line1, = ax.plot(
        df1["elapsed_seconds"],
        df1["memory_gib"],
        label=label1,
        linewidth=2.4,
        color=colors[0],
    )

    if args.file2:
        ax.plot(
            df2["elapsed_seconds"],
            df2["memory_gib"],
            label=label2,
            linewidth=2.4,
            color=colors[1],
        )

    # Peak annotations
    annotate_peak(ax, stats1, colors[0])
    if args.file2:
        annotate_peak(ax, stats2, colors[1])

    # Axis labels
    ax.set_xlabel(
        "Elapsed time [s]",
        fontsize=15,
    )

    ax.set_ylabel(
        f"{'GPU' if args.gpu else 'CPU'} memory usage [GiB]",
        fontsize=15,
    )
    
    # Dynamic y-axis scaling
    ymax = max(stats1["peak"], stats2["peak"]) if args.file2 else stats1["peak"]

    ax.set_ylim(0, 1.30 * ymax)

    # Grid
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.45,
    )

    # Legend
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=12,
        handlelength=1.6,
        borderaxespad=0.0,
    )

    # Tick formatting
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )

    # Metadata box
    stats_text = (
        f"{label1:<30}"
        f"Peak: {stats1['peak']:.2f} GiB   "
        f"Avg: {stats1['avg']:.2f} GiB"
        + (
            f"\n{label2:<30}"
            f"Peak: {stats2['peak']:.2f} GiB   "
            f"Avg: {stats2['avg']:.2f} GiB\n"
            f"{'Relative peak diff':<30}{rel_diff:+.2f}%"
            if args.file2 else ""
        )
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10.5,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            alpha=0.92,
            edgecolor="0.75",
        ),
    )

    # Optional title
    if args.title:
        ax.set_title(
            args.title,
            fontsize=16,
            pad=16,
        )

    # CMS label
    hep.cms.label(
        "",
        data=True,
        com=14,
        ax=ax,
        fontsize=18,
        loc=0,
        pad=0.04,
    )

    # Supplementary label (Simulation / PR Testing / Internal ...)
    ax.text(
        0.11,
        1.005,
        args.cms_label,
        transform=ax.transAxes,
        fontsize=15,
        style="italic",
        va="bottom",
        ha="left",
    )
    
    # Layout
    plt.tight_layout()

    # Save outputs
    save_outputs(fig, args.output, args.gpu)

    # Interactive display
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMS-style hardware memory comparison plotter"
    )

    parser.add_argument(
        "file1",
        help="First input CSV file",
    )

    parser.add_argument(
        "file2",
        nargs="?",
        help="Second input CSV file (optional)",
    )

    parser.add_argument(
        "--label1",
        help="Legend label for first file "
             "(default: filename stem)",
    )

    parser.add_argument(
        "--label2",
        help="Legend label for second file "
             "(default: filename stem)",
    )

    parser.add_argument(
        "--title",
        help="Optional plot title",
    )

    parser.add_argument(
        "--output",
        default="memory_comparison",
        help=(
            "Output filename base "
            "(without extension)"
        ),
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Make CPU or GPU memory profile",
    )
    
    parser.add_argument(
        "--cms-label",
        default="cmssw integration testing",
        help="CMS label text",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display",
    )

    args = parser.parse_args()

    main(args)
