######################################################################
## File: public_plots_tools.py
######################################################################

import os
import math
from colorsys import hls_to_rgb, rgb_to_hls

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib._png import read_png
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox

import numpy as np

######################################################################

FONT_PROPS_SUPTITLE = FontProperties(size="x-large", weight="bold", stretch="condensed")
FONT_PROPS_TITLE = FontProperties(size="large", weight="regular")
FONT_PROPS_AX_TITLE = FontProperties(size="x-large", weight="bold")
FONT_PROPS_TICK_LABEL = FontProperties(size="large", weight="bold")

######################################################################

def InitMatplotlib():
    """Just some Matplotlib settings."""
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["legend.numpoints"] = 1
    matplotlib.rcParams["figure.figsize"] = (8., 6.)
    matplotlib.rcParams["figure.dpi"] = 200
    matplotlib.rcParams["savefig.dpi"] = matplotlib.rcParams["figure.dpi"]
    # End of InitMatplotlib().

######################################################################

def AddLogo(logo_name, ax, zoom=1.2):
    """Read logo from PNG file and add it to axes."""

    logo_data = read_png(logo_name)
    fig_dpi = ax.get_figure().dpi
    fig_size = ax.get_figure().get_size_inches()
    # NOTE: This scaling is kinda ad hoc...
    zoom_factor = .1 / 1.2 * fig_dpi * fig_size[0] / np.shape(logo_data)[0]
    zoom_factor *= zoom
    logo_box = OffsetImage(logo_data, zoom=zoom_factor)
    ann_box = AnnotationBbox(logo_box, [0., 1.],
                             xybox=(2., -3.),
                             xycoords="axes fraction",
                             boxcoords="offset points",
                             box_alignment=(0., 1.),
                             pad=0., frameon=False)
    ax.add_artist(ann_box)
    # End of AddLogo().

######################################################################

def RoundAwayFromZero(val):

    res = None
    if val < 0.:
        res = math.floor(val)
    else:
        res = math.ceil(val)

    # End of RoundAwayFromZero().
    return res

######################################################################

def LatexifyUnits(units_in):

    latex_units = {
        "b^{-1}" : "$\mathbf{b}^{-1}$",
        "mb^{-1}" : "$\mathbf{mb}^{-1}$",
        "ub^{-1}" : "$\mu\mathbf{b}^{-1}$",
        "nb^{-1}" : "$\mathbf{nb}^{-1}$",
        "pb^{-1}" : "$\mathbf{pb}^{-1}$",
        "fb^{-1}" : "$\mathbf{fb}^{-1}$",
        "Hz/b" : "$\mathbf{Hz/b}$",
        "Hz/mb" : "$\mathbf{Hz/mb}$",
        "Hz/ub" : "$\mathbf{Hz/}\mathbf{\mu}\mathbf{b}$",
        "Hz/nb" : "$\mathbf{Hz/nb}$",
        "Hz/pb" : "$\mathbf{Hz/pb}$",
        "Hz/fb" : "$\mathbf{Hz/fb}$"
        }

    res = latex_units[units_in]

    # End of LatexifyUnits().
    return res

######################################################################

def DarkenColor(color_in):
    """Takes a tuple (r, g, b) as input."""

    color_tmp = matplotlib.colors.colorConverter.to_rgb(color_in)

    tmp = rgb_to_hls(*color_tmp)
    color_out = hls_to_rgb(tmp[0], .7 * tmp[1], tmp[2])

    # End of DarkenColor().
    return color_out

######################################################################

class ColorScheme(object):
    """A bit of a cludge, but a simple way to store color choices."""

    @classmethod
    def InitColors(cls):

        #------------------------------
        # For color scheme 'Greg'.
        #------------------------------

        # This is the light blue of the CMS logo.
        ColorScheme.cms_blue = (0./255., 152./255., 212./255.)

        # This is the orange from the CMS logo.
        ColorScheme.cms_orange = (241./255., 194./255., 40./255.)

        # Slightly darker versions of the above colors for the lines.
        ColorScheme.cms_blue_dark = (102./255., 153./255., 204./255.)
        ColorScheme.cms_orange_dark = (255./255., 153./255., 0./255.)

        #------------------------------
        # For color scheme 'Joe'.
        #------------------------------

        # Several colors from the alternative CMS logo, with their
        # darker line variants.

        ColorScheme.cms_red = (208./255., 0./255., 37./255.)
        ColorScheme.cms_yellow = (255./255., 248./255., 0./255.)
        ColorScheme.cms_purple = (125./255., 16./255., 123./255.)
        ColorScheme.cms_green = (60./255., 177./255., 110./255.)
        ColorScheme.cms_orange2 = (227./255., 136./255., 36./255.)

        # End of InitColors().

    def __init__(self, name):

        self.name = name

        # Some defaults.
        self.color_fill_del = "black"
        self.color_fill_rec = "white"
        self.color_fill_peak = "black"
        self.color_line_del = DarkenColor(self.color_fill_del)
        self.color_line_rec = DarkenColor(self.color_fill_rec)
        self.color_line_peak = DarkenColor(self.color_fill_peak)
        self.color_by_year = {
            2010 : "green",
            2011 : "red",
            2012 : "blue"
            }
        self.color_line_pileup = "black"
        self.color_fill_pileup = "blue"
        self.logo_name = "cms_logo_1.png"
        self.file_suffix = "_%s" % self.name.lower()

        tmp_name = self.name.lower()
        if tmp_name == "greg":
            # Color scheme 'Greg'.
            self.color_fill_del = ColorScheme.cms_blue
            self.color_fill_rec = ColorScheme.cms_orange
            self.color_fill_peak = ColorScheme.cms_orange
            self.color_line_del = DarkenColor(self.color_fill_del)
            self.color_line_rec = DarkenColor(self.color_fill_rec)
            self.color_line_peak = DarkenColor(self.color_fill_peak)
            self.color_line_pileup = "black"
            self.color_fill_pileup = ColorScheme.cms_blue
            self.logo_name = "cms_logo_2.png"
            self.file_suffix = ""
        elif tmp_name == "joe":
            # Color scheme 'Joe'.
            self.color_fill_del = ColorScheme.cms_yellow
            self.color_fill_rec = ColorScheme.cms_red
            self.color_fill_peak = ColorScheme.cms_red
            self.color_line_del = DarkenColor(self.color_fill_del)
            self.color_line_rec = DarkenColor(self.color_fill_rec)
            self.color_line_peak = DarkenColor(self.color_fill_peak)
            self.color_line_pileup = "black"
            self.color_fill_pileup = ColorScheme.cms_yellow
            self.logo_name = "cms_logo_3.png"
            self.file_suffix = "_alt"
        else:
            print >> sys.stderr, \
                  "ERROR Unknown color scheme '%s'" % self.name
            sys.exit(1)

        # Find the full path to the logo PNG file.
        # NOTE: This is a little fragile, I think.
        logo_path = os.path.realpath(os.path.dirname(__file__))
        self.logo_name = os.path.join(logo_path,
                                      "../plotdata/%s" % self.logo_name)

        # End of __init__().

    # End of class ColorScheme.

######################################################################
