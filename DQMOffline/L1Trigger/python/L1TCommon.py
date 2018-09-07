def generateEfficiencyStrings(variables, plots):
    stringTemplate = "{plot} " + \
        "'{var} efficiency; Offline E_{{T}}^{{miss}} (GeV); {var} efficiency'" + \
        " {num_path} {den_path}"
    for variable, thresholds in variables.iteritems():
        for plot in plots[variable]:
            for threshold in thresholds:
                plotName = '{0}_threshold_{1}'.format(plot, threshold)
                varName = plot.replace('efficiency', '')
                yield stringTemplate.format(
                    var=varName,
                    plot=plotName,
                    num_path='efficiency_raw/' + plotName + '_Num',
                    den_path='efficiency_raw/' + plotName + '_Den',
                )
