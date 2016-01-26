import urllib, re, json, socket

"""
Python object that enables connection to RR v3 API.
Errors (not bugs!) and more welcome fixes/suggestions please 
post to https://savannah.cern.ch/projects/cmsrunregistry/
"""

class RRApiError(Exception):
    """
    API Exception class
    """

    def __init__(self, resp):
        """
        Construct exception by providing response object.
        """
        if type(resp) == str:
            self.message = resp
        else:
            self.url = resp.geturl()
            self.code = resp.getcode()
            self.stack = None
            for line in resp.read().split("\n"):
                if self.stack == None:
                    m = re.search("<pre>(.*)", line)
                    if m != None:
                        self.stack = m.group(1)
                        m = re.search("^.+\.([^\.]+: .*)$", self.stack)
                        if m != None:
                            self.message = m.group(1)
                        else:
                            self.message = line
                else:
                    m = re.search("(.*)</pre>", line)
                    if m != None:
                        self.stack = self.stack + "\n" + m.group(1)
                        break
                    else:
                        self.stack = self.stack + "\n" + line

    def __str__(self):
        """ Get message """
        return self.message

class RRApi:
    """
    RR API object
    """

    def __init__(self, url, debug = False):
        """
        Construct API object.
        url: URL to RRv3 API, i.e. http://localhost:8080/rr_user
        debug: should debug messages be printed out? Verbose!
        """
        self.debug = debug
        self.url = re.sub("/*$", "/api/", url)
        self.app = self.get(["app"])
        self.dprint("app = ", self.app)

    def dprint(self, *args):
        """
        Print debug information
        """
        if self.debug: 
            print "RRAPI:",
            for arg in args:
                print arg, 
            print

    def get(self, parts, data = None):
        """
        General API call (do not use it directly!)
        """

        #
        # Constructing request path
        #

        callurl = self.url + "/".join(urllib.quote(p) for p in parts)

        #
        # Constructing data payload
        #

        sdata = None
        if data != None:
            sdata = json.dumps(data)

        #
        # Do the query and respond
        #

        self.dprint(callurl, "with payload", sdata)

        resp = urllib.urlopen(callurl, sdata)

        has_getcode = "getcode" in dir(resp)
        if self.debug: 
            if has_getcode:
                self.dprint("Response", resp.getcode(), " ".join(str(resp.info()).split("\r\n")))
            else:
                self.dprint("Response", " ".join(str(resp.info()).split("\r\n")))

        if not has_getcode or resp.getcode() == 200:
            rdata = resp.read()
            if re.search("json", resp.info().gettype()):
                try:
                    return json.loads(rdata)
                except TypeError, e:
                    self.dprint(e)
                    return rdata
            else:
                return rdata
        else:
            raise RRApiError(resp)

    def tags(self):
        """
        Get version tags (USER app only)
        """
        if self.app != "user":
            raise RRApiError("Tags call is possible only in user app")
        return self.get(["tags"])

    def workspaces(self):
        """
        Get workspaces (all apps)
        """
        return self.get(["workspaces"])

    def tables(self, workspace):
        """
        Get tables for workspace (all apps)
        """
        return self.get([workspace, "tables"])

    def columns(self, workspace, table):
        """
        Get columns for table for workspace (all apps)
        """
        return self.get([workspace, table, "columns"])

    def templates(self, workspace, table):
        """
        Get output templates for table for workspace (all apps)
        """
        return self.get([workspace, table, "templates"])

    def count(self, workspace, table, filter = None, query = None, tag = None):
        """
        Get number of rows for table for workspace with filter, query (all apps) or tag (USER app only)
        """

        #
        # Constructing request path
        #

        req = [ workspace, table ]
        if tag != None:
            if self.app != "user":
                raise RRApiError("Tags are possible only in user app")
            else:
                req.append(tag)
        req.append("count")

        #
        # Constructing filter/query payload
        #

        filters = {}
        if filter != None:
            filters['filter'] = filter
        if query != None:
            filters['query'] = query

        return int(self.get(req, filters))

    def data(self, workspace, table, template, columns = None, filter = None, query = None, order = None, tag = None):
        """
        Get data for table for workspace with filter, query (all apps) or tag (USER app only)
        """

        #
        # Check req parameters
        #

        if type(workspace) != str:
            raise RRApiError("workspace parameter must be str")

        #
        # Constructing request path
        #

        req = [ workspace, table, template ]
        if columns != None:
            req.append(",".join(columns))
        else:
            req.append("all")
        if order != None:
            req.append(",".join(order))
        else:
            req.append("none")
        if tag != None:
            if self.app != "user":
                raise RRApiError("Tags are possible only in user app")
            else:
                req.append(tag)
        req.append("data")

        #
        # Constructing filter/query payload
        #

        filters = {}
        if filter != None:
            filters['filter'] = filter
        if query != None:
            filters['query'] = query

        return self.get(req, filters)

    def reports(self, workspace):
        """
        Get available reports (USER app only)
        """
        if self.app != "user":
            raise RRApiError("Reports available only in user app")
        return self.get([workspace, "reports"])
    
    def report(self, workspace, report):
        """
        Get report data (USER app only)
        """
        if self.app != "user":
            raise RRApiError("Reports available only in user app")
        return self.get([workspace, report, "data"])


if __name__ == '__main__':

    print "RR API library."
