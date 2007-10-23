class Node(object):
    def __init__( self ):
        """__init__
        """
        self.__dict__.setdefault('nodeid',0)
        self.__dict__.setdefault('nodelabel','ROOT')
        self.__dict__.setdefault('lft',0)
        self.__dict__.setdefault('rgt',0)
        self.__dict__.setdefault('parentid',0)
        self.__dict__.setdefault('globalsince',0)
        self.__dict__.setdefault('globaltill',0)
        self.__dict__.setdefault('tagid',0)
       # self.__dict__.setdefault('comment','')
    def __repr__( self ):
        """__repr__
        """
        return self.__dict__.__repr__()
    def __setattr__( self, name, value ):
        if not name in self.__dict__:
            raise AttributeError("Unknown attribute "+name)
        self.__dict__[name]=value
    def __getattr__( self, name ):
        if not name in self.__dict__:
            raise AttributeError("Unknown attribute "+name)
        return self.__dict__[name]
    def empty( self ):
        if self.__dict__['nodeid']==0:
            return True
        return False
class LeafNode(Node):
    """The leaf node
    """
    def __init__( self ):
        """__init__
        """
        super(Node,self).__init__()
        super(Node,self).__setattr__( 'tagid',0 )
        self.__dict__.setdefault('tagname','')
        #self.__dict__.setdefault('tagid',0)
        self.__dict__.setdefault('pfn','')
        self.__dict__.setdefault('recordname','')
        self.__dict__.setdefault('objectname','')
        self.__dict__.setdefault('labelname','')
        self.__dict__.setdefault('timetype','')
        self.__dict__.setdefault('comment','')
    def __repr__( self ):
        """__repr__
        """
        if self.tagname=='':
            return ''
        return self.__dict__.__repr__()
if __name__ == "__main__":
    node=Node()
    node.nodeid=1
    print node.nodeid
    print node.nodelabel
    print node.lft
    try:
        node.foreign='a'
    except AttributeError:
        print 'caught right exception'
    except Exception, er:
        print 'unexpected error'
        print str(er)
    leaf=LeafNode()
    print leaf.__class__.__name__,'isLeaf',str(leaf.tagid)
    print node.__class__.__name__,'isLeaf',str(node.tagid)
