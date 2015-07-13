# email: cmsdoxy@cern.ch, ali.mehmet.altundag@cern.ch

# this script generates main pages for CMSSW Refman by using various sources
# such as, doxygen generated html files, persons (work on CMSSW) and their
# email details.. as it is stated in other parsers, in future,  you may need
# to change html tag/attr names depending on output of new doxygen version.
# this script needs:
#  + index.html   : will be used as a template file.
#                 : keep in mind that, this file is source of the doc/html
#                   path source. please see how I set the htmlFilePath var.
#  + files.html   : source of interface files
#  + pages.html   : to get package documentation links
#  + classes.html : to get documentation page links

import sys, os, urllib2, copy
from BeautifulSoup import *
try: import json
except ImportError: import simplejson as json

htmlFullPath = None
htmlFilePath = None
htmlFileName = None
htmlPage     = None
# contentTmplOrg: we need to keep original html source to fix BeautifulSoup
# script tag bug. The problem is that when we edit something by using
# BeautifulSoup, we are not able to play with eddited tags -this can be seen
# as another bug... Please have a look at the bsBugFix function to understand
# why wee need to keep the content html file. --note that this is really
# sensetive approach, you may need to edit something in this python file if
# you change something...
contentTmplOrg = None
contentTmpl  = None
dataSrc      = 'http://cmsdoxy.web.cern.ch/cmsdoxy/cmssw/cmssw.php?ver='
githubBase   = 'https://github.com/cms-sw/cmssw/tree/{0}/{1}'
data         = None
cmsswVersion = None
# tree view template
treeViewTmpl = None

def getFiles(filesPagePath):
    data = {}
    # read and parse files.html to get the file hierarchy
    with open(filesPagePath) as f: page = f.read()
    page = BeautifulSoup(page)
    # please have a look at the files.html page to understand the approach.
    # in short, we use number of '_' character in the id attr to read the
    # file hierarchy.
    table     = page.find('table', {'class' : 'directory'})
    level     = 0
    path      = []
    for row in table.findAll('tr'):
        # first cell is the cell where the info is stored
        id   = row['id']; cell = row.find('td') 
        text = cell.text; url  = '../' + cell.find('a')['href']
        currentLevel = id.count('_')
        # if current level is more than old one, push current item
        if currentLevel > level:
            path.append(text)
        # if current level equals to old one, pop anmd push (replace)
        elif currentLevel == level:
            path.pop(len(path) - 1)
            path.append(text)
        else:
        # if current level is less than old one, pop all items to blance
        # the level. 'plus one' in the loop is to replace last item
            for i in range(level - currentLevel + 1):
                path.pop(len(path) - 1)
            path.append(text)
        level = id.count('_')
        # skip files which are not interface
        if not 'interface' in path: continue
        # no need to have 'interface' node on the tree
        pathWithoutInterface = copy.copy(path)
        pathWithoutInterface.remove('interface')
        # conver the path into tree structure
        node = data
        for i in pathWithoutInterface:
            if not node.has_key(i):
                node[i] = {}
            node = node[i]
    return data

def getPackages(packagesPagePath):
    data = {}
    with open(packagesPagePath) as f: page = f.read()
    page  = BeautifulSoup(page)
    table = page.find('table', {'class' : 'directory'})
    for row in table.findAll('tr'):
        cell = row.find('td')
        url  = '../' + cell.find('a')['href']
        # yeah, it is not that good method to parse a string but it is
        # simple... please see the pages.html file.
        pkg = cell.text.replace('Package ', '').split('/')
        if not data.has_key(pkg[0]): data[pkg[0]] = {}
        if len(pkg) == 2: data[pkg[0]][pkg[1]] = url
        else: data[pkg[0]][pkg[0]] = url
    return data

def getClasses(classesPagePath):
    data = {}
    with open(classesPagePath) as f: page = f.read()
    page  = BeautifulSoup(page)
    content = page.find('div', {'class' : 'contents'})
    for cell in content.findAll('td'):
        aTag = cell.find('a')
        if not aTag or not aTag.has_key('href'): continue
        data[aTag.text] = '../' + aTag['href']
    return data

def prepareTemplate():
    # please notice the fllowing hard coded tags and class names, you may need
    # to change them in future if doxygen changes its html output structure
    header  = htmlPage.find('div', {'class' : 'header'})
    content = htmlPage.find('div', {'class' : 'contents'})

    for tag in header.findAll():
        tag.extract()
    for tag in content.findAll():
        tag.extract()

def costumFormatter(inp):
    if inp.find_parent("script") is None: return EntitySubstitution.substitute_html(inp) 
    else: return inp

def bsBugFix():
    # this function fixes script tag bug of beautifulsoup (bs). bs is escaping
    # javascript operators according to the html escape characters, such as
    # > -> "&gt;". The method to ged rid of this issue is to replace script
    # tags with their original versions in the string level
    html = str(htmlPage)
    for scriptTag in BeautifulSoup(contentTmplOrg).findAll('script'):
        js = scriptTag.text
        html = html.replace(str(scriptTag), '<script>%s</script>' % js)
    return html

def fillContentTemplate(domains):
    rows = ''
    rowTmpl = '<tr id="{0}"><td width="50%">{1}</td><td>{2}</td></tr>'
    aTmpl = """<tr style="padding:0"><td colspan="2" style="padding:0">
               <div class="accordion" id="{0}">
               <iframe width="100%" height="250px" frameborder="0" 
               data-src="iframes/{0}.html"> </iframe>
               </div></td></tr>"""
    domainNames = domains.keys()
    domainNames.sort()
    for domain in domainNames:
        persons = domains[domain].keys()
        persons.sort() 
        cCell = ''
        for person in persons:
            email = domains[domain][person]
            cCell = cCell+'<a href="mailto:{0}">{0}<a/>, '.format(person,email)
        cCell = cCell.rstrip(', ')
        escapedDomainName = domain.replace(' ', '')
        rows  = rows + rowTmpl.format(escapedDomainName, domain, cCell)
        rows  = rows + aTmpl.format(escapedDomainName)
    contentTmpl.find('table').append(BeautifulSoup(rows))
    # put cmssw version
    contentTmpl.find('h2', {'id' : 'version'}).append(cmsswVersion)
    content = htmlPage.find('div', {'class' : 'contents'})
    content.append(contentTmpl)

def generateTree(tree):
    if type(tree) == dict and len(tree) == 0: return BeautifulSoup('')
    # our recursive function to generate domain tree views
    root = BeautifulSoup('<ul></ul>')
    names = tree.keys(); names.sort()
    for name in names:
        node = BeautifulSoup('<li><div></div></li>')
        if type(tree[name]) == dict:
            title = BeautifulSoup('<span class="folder"></span>')
            title.span.append(name)
            node.li.append(title)
            # __git__ and __packageDoc__ are special keys which address links,
            # github and packade documentation links. please see in the section
            # that we merge all what we have (under the __main__ block)
            for i in ['__git__', '__packageDoc__']:
                if not i in tree[name]: continue
                link = BeautifulSoup(' <a></a>')
                link.a['target'] = '_blank'
                link.a['href']   = tree[name][i]
                link.a.append('[%s]' % i.replace('_', ''))
                del tree[name][i]
                title.span.append(link)
            if len(tree[name]) == 0:
                title.span['class'] = 'emptyFolder'
            else: node.li.div['class'] = 'hitarea expandable-hitarea'
            node.li.append(generateTree(tree[name]))
        elif type(tree[name]) == str or type(tree[name]) == unicode:
            link = BeautifulSoup('<a><span class="file"></span></a>')
            link.a['target'] = '_blank'
            link.a['href']   = tree[name]
            link.a.span.append(name)
            node.li.append(link)
        else:
            node.li.append(name)
        root.ul.append(node)
    return root

def generateTreeViewPage(tree, name):
    page = BeautifulSoup(treeViewTmpl)
    treeTag = page.find('ul', {'id' : 'browser'})
    treeTag.append(generateTree(tree))
    twikiLink = page.find('a', {'id' : 'twiki'})
    if name in data['TWIKI_PAGES']:
        twikiLink['href'] = data['TWIKI_PAGES'][name]
    else:
        twikiLink.extract()
    return page
    

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.stderr.write("not enough parameter!\n")
        sys.stderr.write("first pram must be full path of index.html page\n")
        sys.stderr.write("second pram must be full path of index template\n")
        sys.stderr.write("third one must be the CMSSW version\n")
        sys.exit(1)

    htmlFullPath = sys.argv[1]
    # load index.html content template
    with open('%s/IndexContentTemplate.html' % sys.argv[2]) as f:
        contentTmplOrg = f.read()
    with open('%s/TreeViewTemplate.html' % sys.argv[2]) as f:
        treeViewTmpl = f.read()
    contentTmpl  = BeautifulSoup(contentTmplOrg)
    dataSrc      = dataSrc + sys.argv[3]
    htmlFilePath = os.path.split(htmlFullPath)[0]
    htmlFileName = os.path.split(htmlFullPath)[1]
    cmsswVersion = sys.argv[3]

    # load html page
    with open(htmlFullPath) as f: htmlPage = BeautifulSoup(f.read())

    # get json data from cmsdoxy/CMSSWTagCollector
    successFlag = False; loopLimit = 3
    while(not successFlag and loopLimit > 0):
        loopLimit = loopLimit - 1
        try:
            print 'reading data from cmsdoxy/CMSSWTagCollector...'
            data = urllib2.urlopen(dataSrc).read()
            data = json.loads(data)
            successFlag = True
        except:
            print 'I couldn\'t get the data. Trying again...'
    # if you cannot get data from the CMSSWTagCollector,
    # inform user and exit
    if not successFlag:
        sys.stderr.write("I couldn't get the data from %s\n" % dataSrc)
        sys.stderr.write("I am not able to generate the main page, ")
        sys.stderr.write("I will leave it as it is...\n")
        sys.stderr.write("# PLEASE SEND AN EMAIL TO cmsdoxy[at]cern.ch\n")
        sys.exit(1)

    print 'parsing source file hierarchy...'
    files    = getFiles("%s/files.html" % htmlFilePath)

    print 'parsing packages...'
    packages = getPackages('%s/pages.html' % htmlFilePath)

    print 'parsing classes...'
    classes  = getClasses("%s/classes.html" % htmlFilePath)

    tree = copy.copy(data['CMSSW_CATEGORIES'])
    print "generating tree views..."
    # merge files and the tree collected from cmsdoxy/CMSSWTagCollector
    for domain in tree: # Core
        for l1 in tree[domain]: # Configuration
            for l2 in tree[domain][l1]:
                # put github link if exists in classes dict
                link = githubBase.format(cmsswVersion, '%s/%s'%(l1,l2))
                tree[domain][l1][l2]['__git__'] = link
                # prepare package documentation link if exits 
                if packages.has_key(l1) and packages[l1].has_key(l2):
                    tree[domain][l1][l2]['__packageDoc__'] = packages[l1][l2]
                if not l1 in files or not l2 in files[l1]: continue
                for file in files[l1][l2]:
                    # no need to have header file extension (.h)
                    file = file.replace('.h', '')
                    if not file in tree[domain][l1][l2]:
                        tree[domain][l1][l2] = {}
                    if file in classes:
                        tree[domain][l1][l2][file] = classes[file]
                    else:
                        tree[domain][l1][l2][file] = None

    # we got the data from cmsdoxy/CMSSWTagCollector, we can start prapering
    # the html main page now.
    prepareTemplate()

    print "generating mainpage..."
    fillContentTemplate(data['PERSON_MAP'])

    with open("%s/index.html" % htmlFilePath, 'w') as f:
        f.write(bsBugFix())

    print 'generating tree views...'
    # generate tree view pages
    for domain in tree:
        page  = generateTreeViewPage(tree[domain], domain)
        fName = domain.replace(' ', '')
        with open('%s/iframes/%s.html' % (htmlFilePath, fName), 'w') as f:
            f.write(str(page))
