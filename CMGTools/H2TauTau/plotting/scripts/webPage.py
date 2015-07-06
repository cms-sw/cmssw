import shutil
import glob
import pprint

def mkdir(dir):
    adir = '/'.join([os.getcwd(), dir])
    if os.path.isdir(dir):
        while 1:
            answer = raw_input(adir + ' exists. Do you want to remove it? [y/n]')
            if answer=='y':
                os.system('rm -r '+dir)
                break
            elif answer=='n':
                sys.exit(1)
            else:
                os.system('rm -r '+dir)
                break
    os.mkdir(dir)


def addFile(file, outdir):
    newname = '/'.join([outdir, os.path.basename(file)])
    shutil.copyfile(file, newname )
    return newname


def addCategory(categ, outdir):
    # print categ
    # adding a directory for this category in outdir
    codir = '/'.join([outdir, categ['name'] ])
    os.mkdir(codir)
    lines = []
    for idx, image in enumerate(categ['images']):
        # print image
        # print codir
        fileName = addFile(image, codir)
        fileName = fileName.split(outdir+'/')[1]
#        if idx%2==0:
#            if idx>0:
#                lines.append('</tr>\n')
#            lines.append('<tr>\n')
        img = '<td width=30%><img src="{name}" alt="{name}" width="100%"></img></td>\n'.format(name=fileName)
        lines.append(img)
    return lines


def addOfficialPlot(categ, blessed):
    # pat = 'HCP-121101/postfit/sm/eleTau_{categ}_unscaled_7TeV_.png'.format(categ=categ['name'] )
    pat = blessed.format(categ=categ['name'])
    ofplots = glob.glob(pat)
    # print 'XXXX ', categ['name']
    # print pat
    # print ofplots
    categ['images'].extend(ofplots)
    return categ
    
def fillTemplate(template, dcdir, outdir, blessed):
    # find all categories, and all plots in these categories
    categories = {}
    for dnam in os.listdir(dcdir):
        fdnam = '/'.join([dcdir, dnam])
        if not os.path.isdir(fdnam):
            continue
        categ = dict(
            name = dnam,
            dir = fdnam,
            images = glob.glob('/'.join([fdnam, '*.png']))
            )
        categ = addOfficialPlot(categ, blessed)
        categories[dnam] = categ
        
    tablestart = False
    out = open('/'.join([outdir, 'index.html']),'w')
    html = open(template)
    for line in html:
        out.write(line)
        if line.startswith('<table'):
            tablestart = True
        if tablestart:
            for categ in categories.values():
                out.write( '<tr><td><h2>{categ}</h2></td><td></td></tr>\n'.format(categ=categ['name']) )
                out.write('<tr>\n')
                ls = addCategory(categ, outdir)
                for l in ls:
                    out.write(l)
                out.write('</tr>\n')
                tablestart = False
    out.close()
    html.close()
    
if __name__ == '__main__':

    import os
    import sys
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog <datacards_dir> <out_dir>
    """

    parser.add_option("-t","--template", dest="template",
                      default='index.html',
                      help='location of your input index.html template')
    parser.add_option("-b","--blessed", dest="blessed",
                      default=None,
                      help="Pattern for blessed official plots. For example:\nHCP-121101/postfit/sm/eleTau_{categ}_unscaled_7TeV_.png")

    
    (options,args) = parser.parse_args()
    
    if len(args)!=2:
        parser.print_usage()
        print 'provide exactly 2 arguments.'
        print args
        sys.exit(1)

    dcdir, outdir = args

    if not os.path.isdir(dcdir):
        print 'input datacard directory does not exit:', dcdir
        sys.exit(1)
    mkdir(outdir)

    if not os.path.isfile(options.template):
        print 'input template does not exit:', options.template
        sys.exit(1)


    fillTemplate(options.template, dcdir, outdir, options.blessed)
    
    # addFile(options.template)
    
    

    
