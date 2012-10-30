import numpy as np
import itertools
import matplotlib.cbook as cbook
iterable = cbook.iterable
is_string_like = cbook.is_string_like
is_sequence_of_strings = cbook.is_sequence_of_strings

def hist(self, x, bins=10, range=None, normed=False, weights=None,
         cumulative=False, bottom=None, histtype='bar', align='mid',
         orientation='vertical', rwidth=None, log=False,
         color=None, label=None,
         **kwargs):
    """
    call signature::

      hist(x, bins=10, range=None, normed=False, cumulative=False,
           bottom=None, histtype='bar', align='mid',
           orientation='vertical', rwidth=None, log=False, **kwargs)

    Compute and draw the histogram of *x*. The return value is a
    tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*,
    [*patches0*, *patches1*,...]) if the input contains multiple
    data.

    Multiple data can be provided via *x* as a list of datasets
    of potentially different length ([*x0*, *x1*, ...]), or as
    a 2-D ndarray in which each column is a dataset.  Note that
    the ndarray form is transposed relative to the list form.

    Masked arrays are not supported at present.

    Keyword arguments:

      *bins*:
        Either an integer number of bins or a sequence giving the
        bins.  If *bins* is an integer, *bins* + 1 bin edges
        will be returned, consistent with :func:`numpy.histogram`
        for numpy version >= 1.3, and with the *new* = True argument
        in earlier versions.
        Unequally spaced bins are supported if *bins* is a sequence.

      *range*:
        The lower and upper range of the bins. Lower and upper outliers
        are ignored. If not provided, *range* is (x.min(), x.max()).
        Range has no effect if *bins* is a sequence.

        If *bins* is a sequence or *range* is specified, autoscaling
        is based on the specified bin range instead of the
        range of x.

      *normed*:
        If *True*, the first element of the return tuple will
        be the counts normalized to form a probability density, i.e.,
        ``n/(len(x)*dbin)``.  In a probability density, the integral of
        the histogram should be 1; you can verify that with a
        trapezoidal integration of the probability density function::

          pdf, bins, patches = ax.hist(...)
          print np.sum(pdf * np.diff(bins))

        .. Note:: Until numpy release 1.5, the underlying numpy
                  histogram function was incorrect with *normed*=*True*
                  if bin sizes were unequal.  MPL inherited that
                  error.  It is now corrected within MPL when using
                  earlier numpy versions

      *weights*
        An array of weights, of the same shape as *x*.  Each value in
        *x* only contributes its associated weight towards the bin
        count (instead of 1).  If *normed* is True, the weights are
        normalized, so that the integral of the density over the range
        remains 1.

      *cumulative*:
        If *True*, then a histogram is computed where each bin
        gives the counts in that bin plus all bins for smaller values.
        The last bin gives the total number of datapoints.  If *normed*
        is also *True* then the histogram is normalized such that the
        last bin equals 1. If *cumulative* evaluates to less than 0
        (e.g. -1), the direction of accumulation is reversed.  In this
        case, if *normed* is also *True*, then the histogram is normalized
        such that the first bin equals 1.

      *histtype*: [ 'bar' | 'barstacked' | 'step' | 'stepfilled' ]
        The type of histogram to draw.

          - 'bar' is a traditional bar-type histogram.  If multiple data
            are given the bars are aranged side by side.

          - 'barstacked' is a bar-type histogram where multiple
            data are stacked on top of each other.

          - 'step' generates a lineplot that is by default
            unfilled.

          - 'stepfilled' generates a lineplot that is by default
            filled.

      *align*: ['left' | 'mid' | 'right' ]
        Controls how the histogram is plotted.

          - 'left': bars are centered on the left bin edges.

          - 'mid': bars are centered between the bin edges.

          - 'right': bars are centered on the right bin edges.

      *orientation*: [ 'horizontal' | 'vertical' ]
        If 'horizontal', :func:`~matplotlib.pyplot.barh` will be
        used for bar-type histograms and the *bottom* kwarg will be
        the left edges.

      *rwidth*:
        The relative width of the bars as a fraction of the bin
        width.  If *None*, automatically compute the width. Ignored
        if *histtype* = 'step' or 'stepfilled'.

      *log*:
        If *True*, the histogram axis will be set to a log scale.
        If *log* is *True* and *x* is a 1D array, empty bins will
        be filtered out and only the non-empty (*n*, *bins*,
        *patches*) will be returned.

      *color*:
        Color spec or sequence of color specs, one per
        dataset.  Default (*None*) uses the standard line
        color sequence.

      *label*:
        String, or sequence of strings to match multiple
        datasets.  Bar charts yield multiple patches per
        dataset, but only the first gets the label, so
        that the legend command will work as expected::

            ax.hist(10+2*np.random.randn(1000), label='men')
            ax.hist(12+3*np.random.randn(1000), label='women', alpha=0.5)
            ax.legend()

    kwargs are used to update the properties of the
    :class:`~matplotlib.patches.Patch` instances returned by *hist*:

    %(Patch)s

    **Example:**

    .. plot:: mpl_examples/pylab_examples/histogram_demo.py
    """
    if not self._hold: self.cla()

    # NOTE: the range keyword overwrites the built-in func range !!!
    #       needs to be fixed in numpy                           !!!

    # Validate string inputs here so we don't have to clutter
    # subsequent code.
    if histtype not in ['bar', 'barstacked', 'step', 'stepfilled']:
        raise ValueError("histtype %s is not recognized" % histtype)

    if align not in ['left', 'mid', 'right']:
        raise ValueError("align kwarg %s is not recognized" % align)

    if orientation not in [ 'horizontal', 'vertical']:
        raise ValueError(
            "orientation kwarg %s is not recognized" % orientation)


    if kwargs.get('width') is not None:
        raise DeprecationWarning(
            'hist now uses the rwidth to give relative width '
            'and not absolute width')

    # Massage 'x' for processing.
    # NOTE: Be sure any changes here is also done below to 'weights'
    if isinstance(x, np.ndarray) or not iterable(x[0]):
        # TODO: support masked arrays;
        x = np.asarray(x)
        if x.ndim == 2:
            x = x.T # 2-D input with columns as datasets; switch to rows
        elif x.ndim == 1:
            x = x.reshape(1, x.shape[0])  # new view, single row
        else:
            raise ValueError("x must be 1D or 2D")
        if x.shape[1] < x.shape[0]:
            warnings.warn('2D hist input should be nsamples x nvariables;\n '
                'this looks transposed (shape is %d x %d)' % x.shape[::-1])
    else:
        # multiple hist with data of different length
        x = [np.array(xi) for xi in x]

    nx = len(x) # number of datasets

    if color is None:
        color = [self._get_lines.color_cycle.next()
                                        for i in xrange(nx)]
    else:
        color = mcolors.colorConverter.to_rgba_array(color)
        if len(color) != nx:
            raise ValueError("color kwarg must have one color per dataset")

    # We need to do to 'weights' what was done to 'x'
    if weights is not None:
        if isinstance(weights, np.ndarray) or not iterable(weights[0]) :
            w = np.array(weights)
            if w.ndim == 2:
                w = w.T
            elif w.ndim == 1:
                w.shape = (1, w.shape[0])
            else:
                raise ValueError("weights must be 1D or 2D")
        else:
            w = [np.array(wi) for wi in weights]

        if len(w) != nx:
            raise ValueError('weights should have the same shape as x')
        for i in xrange(nx):
            if len(w[i]) != len(x[i]):
                raise ValueError(
                    'weights should have the same shape as x')
    else:
        w = [None]*nx


    # Save autoscale state for later restoration; turn autoscaling
    # off so we can do it all a single time at the end, instead
    # of having it done by bar or fill and then having to be redone.
    _saved_autoscalex = self.get_autoscalex_on()
    _saved_autoscaley = self.get_autoscaley_on()
    self.set_autoscalex_on(False)
    self.set_autoscaley_on(False)

    # Save the datalimits for the same reason:
    _saved_bounds = self.dataLim.bounds

    # Check whether bins or range are given explicitly. In that
    # case use those values for autoscaling.
    binsgiven = (cbook.iterable(bins) or range != None)

    # If bins are not specified either explicitly or via range,
    # we need to figure out the range required for all datasets,
    # and supply that to np.histogram.
    if not binsgiven:
        xmin = np.inf
        xmax = -np.inf
        for xi in x:
            xmin = min(xmin, xi.min())
            xmax = max(xmax, xi.max())
        range = (xmin, xmax)

    #hist_kwargs = dict(range=range, normed=bool(normed))
    # We will handle the normed kwarg within mpl until we
    # get to the point of requiring numpy >= 1.5.
    hist_kwargs = dict(range=range)
    if np.__version__ < "1.3": # version 1.1 and 1.2
        hist_kwargs['new'] = True

    n = []
    for i in xrange(nx):
        # this will automatically overwrite bins,
        # so that each histogram uses the same bins
        m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
        if normed:
            db = np.diff(bins)
            m = (m.astype(float) / db) / m.sum()
        n.append(m)
    if normed and db.std() > 0.01 * db.mean():
        warnings.warn("""
        This release fixes a normalization bug in the NumPy histogram
        function prior to version 1.5, occuring with non-uniform
        bin widths. The returned and plotted value is now a density:
            n / (N * bin width),
        where n is the bin count and N the total number of points.
        """)



    if cumulative:
        slc = slice(None)
        if cbook.is_numlike(cumulative) and cumulative < 0:
            slc = slice(None,None,-1)

        if normed:
            n = [(m * np.diff(bins))[slc].cumsum()[slc] for m in n]
        else:
            n = [m[slc].cumsum()[slc] for m in n]

    patches = []

    if histtype.startswith('bar'):
        totwidth = np.diff(bins)

        if rwidth is not None:
            dr = min(1.0, max(0.0, rwidth))
        elif len(n)>1:
            dr = 0.8
        else:
            dr = 1.0

        if histtype=='bar':
            width = dr*totwidth/nx
            dw = width

            if nx > 1:
                boffset = -0.5*dr*totwidth*(1.0-1.0/nx)
            else:
                boffset = 0.0
            stacked = False
        elif histtype=='barstacked':
            width = dr*totwidth
            boffset, dw = 0.0, 0.0
            stacked = True

        if align == 'mid' or align == 'edge':
            boffset += 0.5*totwidth
        elif align == 'right':
            boffset += totwidth

        if orientation == 'horizontal':
            _barfunc = self.barh
        else:  # orientation == 'vertical'
            _barfunc = self.bar

        for m, c in zip(n, color):
            patch = _barfunc(bins[:-1]+boffset, m, width, bottom,
                              align='center', log=log,
                              color=c)
            patches.append(patch)
            if stacked:
                if bottom is None:
                    bottom = 0.0
                bottom += m
            boffset += dw

    elif histtype.startswith('step'):
        x = np.zeros( 2*len(bins), np.float )
        y = np.zeros( 2*len(bins), np.float )

        x[0::2], x[1::2] = bins, bins

        # FIX FIX FIX
        # This is the only real change.
        # minimum = min(bins)
        if log is True:
            minimum = 1.0
        elif log:
            minimum = float(log)
        else:
            minimum = 0.0
        # FIX FIX FIX end

        if align == 'left' or align == 'center':
            x -= 0.5*(bins[1]-bins[0])
        elif align == 'right':
            x += 0.5*(bins[1]-bins[0])

        if log:
            y[0],y[-1] = minimum, minimum
            if orientation == 'horizontal':
                self.set_xscale('log')
            else:  # orientation == 'vertical'
                self.set_yscale('log')

        fill = (histtype == 'stepfilled')

        for m, c in zip(n, color):
            y[1:-1:2], y[2::2] = m, m
            if log:
                y[y<minimum]=minimum
            if orientation == 'horizontal':
                x,y = y,x

            if fill:
                patches.append( self.fill(x, y,
                    closed=False, facecolor=c) )
            else:
                patches.append( self.fill(x, y,
                    closed=False, edgecolor=c, fill=False) )

        # adopted from adjust_x/ylim part of the bar method
        if orientation == 'horizontal':
            xmin0 = max(_saved_bounds[0]*0.9, minimum)
            xmax = self.dataLim.intervalx[1]
            for m in n:
                xmin = np.amin(m[m!=0]) # filter out the 0 height bins
            xmin = max(xmin*0.9, minimum)
            xmin = min(xmin0, xmin)
            self.dataLim.intervalx = (xmin, xmax)
        elif orientation == 'vertical':
            ymin0 = max(_saved_bounds[1]*0.9, minimum)
            ymax = self.dataLim.intervaly[1]
            for m in n:
                ymin = np.amin(m[m!=0]) # filter out the 0 height bins
            ymin = max(ymin*0.9, minimum)
            ymin = min(ymin0, ymin)
            self.dataLim.intervaly = (ymin, ymax)

    if label is None:
        labels = ['_nolegend_']
    elif is_string_like(label):
        labels = [label]
    elif is_sequence_of_strings(label):
        labels = list(label)
    else:
        raise ValueError(
            'invalid label: must be string or sequence of strings')
    if len(labels) < nx:
        labels += ['_nolegend_'] * (nx - len(labels))

    for (patch, lbl) in zip(patches, labels):
        for p in patch:
            p.update(kwargs)
            p.set_label(lbl)
            lbl = '_nolegend_'

    if binsgiven:
        if orientation == 'vertical':
            self.update_datalim([(bins[0],0), (bins[-1],0)], updatey=False)
        else:
            self.update_datalim([(0,bins[0]), (0,bins[-1])], updatex=False)

    self.set_autoscalex_on(_saved_autoscalex)
    self.set_autoscaley_on(_saved_autoscaley)
    self.autoscale_view()

    if nx == 1:
        return n[0], bins, cbook.silent_list('Patch', patches[0])
    else:
        return n, bins, cbook.silent_list('Lists of Patches', patches)
