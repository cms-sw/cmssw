webpackHotUpdate_N_E("pages/index",{

/***/ "./node_modules/antd/lib/layout/layout.js":
/*!************************************************!*\
  !*** ./node_modules/antd/lib/layout/layout.js ***!
  \************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = exports.Content = exports.Footer = exports.Header = exports.LayoutContext = void 0;

var _toConsumableArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/toConsumableArray */ "./node_modules/@babel/runtime/helpers/toConsumableArray.js"));

var _defineProperty2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/defineProperty */ "./node_modules/@babel/runtime/helpers/defineProperty.js"));

var _slicedToArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/slicedToArray */ "./node_modules/@babel/runtime/helpers/slicedToArray.js"));

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _classnames = _interopRequireDefault(__webpack_require__(/*! classnames */ "./node_modules/classnames/index.js"));

var _configProvider = __webpack_require__(/*! ../config-provider */ "./node_modules/antd/lib/config-provider/index.js");

var __rest = void 0 && (void 0).__rest || function (s, e) {
  var t = {};

  for (var p in s) {
    if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0) t[p] = s[p];
  }

  if (s != null && typeof Object.getOwnPropertySymbols === "function") for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
    if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i])) t[p[i]] = s[p[i]];
  }
  return t;
};

var LayoutContext = /*#__PURE__*/React.createContext({
  siderHook: {
    addSider: function addSider() {
      return null;
    },
    removeSider: function removeSider() {
      return null;
    }
  }
});
exports.LayoutContext = LayoutContext;

function generator(_ref) {
  var suffixCls = _ref.suffixCls,
      tagName = _ref.tagName,
      displayName = _ref.displayName;
  return function (BasicComponent) {
    var Adapter = function Adapter(props) {
      var _React$useContext = React.useContext(_configProvider.ConfigContext),
          getPrefixCls = _React$useContext.getPrefixCls;

      var customizePrefixCls = props.prefixCls;
      var prefixCls = getPrefixCls(suffixCls, customizePrefixCls);
      return /*#__PURE__*/React.createElement(BasicComponent, (0, _extends2["default"])({
        prefixCls: prefixCls,
        tagName: tagName
      }, props));
    };

    Adapter.displayName = displayName;
    return Adapter;
  };
}

var Basic = function Basic(props) {
  var prefixCls = props.prefixCls,
      className = props.className,
      children = props.children,
      tagName = props.tagName,
      others = __rest(props, ["prefixCls", "className", "children", "tagName"]);

  var classString = (0, _classnames["default"])(prefixCls, className);
  return /*#__PURE__*/React.createElement(tagName, (0, _extends2["default"])({
    className: classString
  }, others), children);
};

var BasicLayout = function BasicLayout(props) {
  var _classNames;

  var _React$useContext2 = React.useContext(_configProvider.ConfigContext),
      direction = _React$useContext2.direction;

  var _React$useState = React.useState([]),
      _React$useState2 = (0, _slicedToArray2["default"])(_React$useState, 2),
      siders = _React$useState2[0],
      setSiders = _React$useState2[1];

  var prefixCls = props.prefixCls,
      className = props.className,
      children = props.children,
      hasSider = props.hasSider,
      Tag = props.tagName,
      others = __rest(props, ["prefixCls", "className", "children", "hasSider", "tagName"]);

  var classString = (0, _classnames["default"])(prefixCls, (_classNames = {}, (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-has-sider"), typeof hasSider === 'boolean' ? hasSider : siders.length > 0), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-rtl"), direction === 'rtl'), _classNames), className);
  return /*#__PURE__*/React.createElement(LayoutContext.Provider, {
    value: {
      siderHook: {
        addSider: function addSider(id) {
          setSiders(function (prev) {
            return [].concat((0, _toConsumableArray2["default"])(prev), [id]);
          });
        },
        removeSider: function removeSider(id) {
          setSiders(function (prev) {
            return prev.filter(function (currentId) {
              return currentId !== id;
            });
          });
        }
      }
    }
  }, /*#__PURE__*/React.createElement(Tag, (0, _extends2["default"])({
    className: classString
  }, others), children));
};

var Layout = generator({
  suffixCls: 'layout',
  tagName: 'section',
  displayName: 'Layout'
})(BasicLayout);
var Header = generator({
  suffixCls: 'layout-header',
  tagName: 'header',
  displayName: 'Header'
})(Basic);
exports.Header = Header;
var Footer = generator({
  suffixCls: 'layout-footer',
  tagName: 'footer',
  displayName: 'Footer'
})(Basic);
exports.Footer = Footer;
var Content = generator({
  suffixCls: 'layout-content',
  tagName: 'main',
  displayName: 'Content'
})(Basic);
exports.Content = Content;
var _default = Layout;
exports["default"] = _default;

/***/ }),

/***/ "./pages/index.tsx":
/*!*************************!*\
  !*** ./pages/index.tsx ***!
  \*************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/head */ "./node_modules/next/head.js");
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_head__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styles/styledComponents */ "./styles/styledComponents.ts");
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _containers_display_header__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../containers/display/header */ "./containers/display/header.tsx");
/* harmony import */ var _containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../containers/display/content/constent_switching */ "./containers/display/content/constent_switching.tsx");
/* harmony import */ var _components_modes_modesSelection__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../components/modes/modesSelection */ "./components/modes/modesSelection.tsx");
/* harmony import */ var antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! antd/lib/layout/layout */ "./node_modules/antd/lib/layout/layout.js");
/* harmony import */ var antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9___default = /*#__PURE__*/__webpack_require__.n(antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9__);
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/pages/index.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;











var Index = function Index() {
  _s();

  // We grab the query from the URL:
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var isDatasetAndRunNumberSelected = !!query.run_number && !!query.dataset_name;
  return __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 5
    }
  }, __jsx(next_head__WEBPACK_IMPORTED_MODULE_1___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 7
    }
  }, __jsx("script", {
    crossOrigin: "anonymous",
    type: "text/javascript",
    src: "./jsroot-5.8.0/scripts/JSRootCore.js?2d&hist&more2d",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 9
    }
  })), __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLayout"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 7
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 40,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    style: {
      display: 'flex',
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Tooltip"], {
    title: "Back to main page",
    placement: "bottomLeft",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 15
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 17
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoWrapper"], {
    onClick: function onClick(e) {
      return Object(_utils_pages__WEBPACK_IMPORTED_MODULE_5__["backToMainPage"])(e);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 45,
      columnNumber: 19
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogo"], {
    src: "./images/CMSlogo_white_red_nolabel_1024_May2014.png",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 21
    }
  })))), __jsx(_components_modes_modesSelection__WEBPACK_IMPORTED_MODULE_8__["ModesSelection"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 14
    }
  }))), __jsx(_containers_display_header__WEBPACK_IMPORTED_MODULE_6__["Header"], {
    isDatasetAndRunNumberSelected: isDatasetAndRunNumberSelected,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 11
    }
  })), __jsx(_containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__["ContentSwitching"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 9
    }
  }), __jsx(antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9__["Footer"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 9
    }
  })));
};

_s(Index, "fN7XvhJ+p5oE6+Xlo0NJmXpxjC8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = Index;
/* harmony default export */ __webpack_exports__["default"] = (Index);

var _c;

$RefreshReg$(_c, "Index");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vbm9kZV9tb2R1bGVzL2FudGQvbGliL2xheW91dC9sYXlvdXQuanMiLCJ3ZWJwYWNrOi8vX05fRS8uL3BhZ2VzL2luZGV4LnRzeCJdLCJuYW1lcyI6WyJJbmRleCIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5IiwiaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwiZGlzcGxheSIsImFsaWduSXRlbXMiLCJlIiwiYmFja1RvTWFpblBhZ2UiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBYTs7QUFFYiw4QkFBOEIsbUJBQU8sQ0FBQyxzSEFBK0M7O0FBRXJGLDZCQUE2QixtQkFBTyxDQUFDLG9IQUE4Qzs7QUFFbkY7QUFDQTtBQUNBLENBQUM7QUFDRDs7QUFFQSxpREFBaUQsbUJBQU8sQ0FBQyw0R0FBMEM7O0FBRW5HLDhDQUE4QyxtQkFBTyxDQUFDLHNHQUF1Qzs7QUFFN0YsNkNBQTZDLG1CQUFPLENBQUMsb0dBQXNDOztBQUUzRix1Q0FBdUMsbUJBQU8sQ0FBQyx3RkFBZ0M7O0FBRS9FLG9DQUFvQyxtQkFBTyxDQUFDLDRDQUFPOztBQUVuRCx5Q0FBeUMsbUJBQU8sQ0FBQyxzREFBWTs7QUFFN0Qsc0JBQXNCLG1CQUFPLENBQUMsNEVBQW9COztBQUVsRDtBQUNBOztBQUVBO0FBQ0E7QUFDQTs7QUFFQSwySEFBMkgsY0FBYztBQUN6STtBQUNBO0FBQ0E7QUFDQTs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLEtBQUs7QUFDTDtBQUNBO0FBQ0E7QUFDQTtBQUNBLENBQUM7QUFDRDs7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxPQUFPO0FBQ1A7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUVBO0FBQ0E7QUFDQTtBQUNBLEdBQUc7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QUFBLElBcEJGLENBUkYsQ0FERjtBQWlDRCxDQXhDRDs7R0FBTUgsSztVQUVXRSxxRDs7O0tBRlhGLEs7QUEwQ1NBLG9FQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjU2YmZjYzdkNGQwYmRiNTM0MjE2LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJcInVzZSBzdHJpY3RcIjtcblxudmFyIF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVXaWxkY2FyZFwiKTtcblxudmFyIF9pbnRlcm9wUmVxdWlyZURlZmF1bHQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZURlZmF1bHRcIik7XG5cbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwge1xuICB2YWx1ZTogdHJ1ZVxufSk7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IGV4cG9ydHMuQ29udGVudCA9IGV4cG9ydHMuRm9vdGVyID0gZXhwb3J0cy5IZWFkZXIgPSBleHBvcnRzLkxheW91dENvbnRleHQgPSB2b2lkIDA7XG5cbnZhciBfdG9Db25zdW1hYmxlQXJyYXkyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy90b0NvbnN1bWFibGVBcnJheVwiKSk7XG5cbnZhciBfZGVmaW5lUHJvcGVydHkyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9kZWZpbmVQcm9wZXJ0eVwiKSk7XG5cbnZhciBfc2xpY2VkVG9BcnJheTIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL3NsaWNlZFRvQXJyYXlcIikpO1xuXG52YXIgX2V4dGVuZHMyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9leHRlbmRzXCIpKTtcblxudmFyIFJlYWN0ID0gX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQocmVxdWlyZShcInJlYWN0XCIpKTtcblxudmFyIF9jbGFzc25hbWVzID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiY2xhc3NuYW1lc1wiKSk7XG5cbnZhciBfY29uZmlnUHJvdmlkZXIgPSByZXF1aXJlKFwiLi4vY29uZmlnLXByb3ZpZGVyXCIpO1xuXG52YXIgX19yZXN0ID0gdm9pZCAwICYmICh2b2lkIDApLl9fcmVzdCB8fCBmdW5jdGlvbiAocywgZSkge1xuICB2YXIgdCA9IHt9O1xuXG4gIGZvciAodmFyIHAgaW4gcykge1xuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocywgcCkgJiYgZS5pbmRleE9mKHApIDwgMCkgdFtwXSA9IHNbcF07XG4gIH1cblxuICBpZiAocyAhPSBudWxsICYmIHR5cGVvZiBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzID09PSBcImZ1bmN0aW9uXCIpIGZvciAodmFyIGkgPSAwLCBwID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhzKTsgaSA8IHAubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoZS5pbmRleE9mKHBbaV0pIDwgMCAmJiBPYmplY3QucHJvdG90eXBlLnByb3BlcnR5SXNFbnVtZXJhYmxlLmNhbGwocywgcFtpXSkpIHRbcFtpXV0gPSBzW3BbaV1dO1xuICB9XG4gIHJldHVybiB0O1xufTtcblxudmFyIExheW91dENvbnRleHQgPSAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlQ29udGV4dCh7XG4gIHNpZGVySG9vazoge1xuICAgIGFkZFNpZGVyOiBmdW5jdGlvbiBhZGRTaWRlcigpIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH0sXG4gICAgcmVtb3ZlU2lkZXI6IGZ1bmN0aW9uIHJlbW92ZVNpZGVyKCkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICB9XG59KTtcbmV4cG9ydHMuTGF5b3V0Q29udGV4dCA9IExheW91dENvbnRleHQ7XG5cbmZ1bmN0aW9uIGdlbmVyYXRvcihfcmVmKSB7XG4gIHZhciBzdWZmaXhDbHMgPSBfcmVmLnN1ZmZpeENscyxcbiAgICAgIHRhZ05hbWUgPSBfcmVmLnRhZ05hbWUsXG4gICAgICBkaXNwbGF5TmFtZSA9IF9yZWYuZGlzcGxheU5hbWU7XG4gIHJldHVybiBmdW5jdGlvbiAoQmFzaWNDb21wb25lbnQpIHtcbiAgICB2YXIgQWRhcHRlciA9IGZ1bmN0aW9uIEFkYXB0ZXIocHJvcHMpIHtcbiAgICAgIHZhciBfUmVhY3QkdXNlQ29udGV4dCA9IFJlYWN0LnVzZUNvbnRleHQoX2NvbmZpZ1Byb3ZpZGVyLkNvbmZpZ0NvbnRleHQpLFxuICAgICAgICAgIGdldFByZWZpeENscyA9IF9SZWFjdCR1c2VDb250ZXh0LmdldFByZWZpeENscztcblxuICAgICAgdmFyIGN1c3RvbWl6ZVByZWZpeENscyA9IHByb3BzLnByZWZpeENscztcbiAgICAgIHZhciBwcmVmaXhDbHMgPSBnZXRQcmVmaXhDbHMoc3VmZml4Q2xzLCBjdXN0b21pemVQcmVmaXhDbHMpO1xuICAgICAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KEJhc2ljQ29tcG9uZW50LCAoMCwgX2V4dGVuZHMyW1wiZGVmYXVsdFwiXSkoe1xuICAgICAgICBwcmVmaXhDbHM6IHByZWZpeENscyxcbiAgICAgICAgdGFnTmFtZTogdGFnTmFtZVxuICAgICAgfSwgcHJvcHMpKTtcbiAgICB9O1xuXG4gICAgQWRhcHRlci5kaXNwbGF5TmFtZSA9IGRpc3BsYXlOYW1lO1xuICAgIHJldHVybiBBZGFwdGVyO1xuICB9O1xufVxuXG52YXIgQmFzaWMgPSBmdW5jdGlvbiBCYXNpYyhwcm9wcykge1xuICB2YXIgcHJlZml4Q2xzID0gcHJvcHMucHJlZml4Q2xzLFxuICAgICAgY2xhc3NOYW1lID0gcHJvcHMuY2xhc3NOYW1lLFxuICAgICAgY2hpbGRyZW4gPSBwcm9wcy5jaGlsZHJlbixcbiAgICAgIHRhZ05hbWUgPSBwcm9wcy50YWdOYW1lLFxuICAgICAgb3RoZXJzID0gX19yZXN0KHByb3BzLCBbXCJwcmVmaXhDbHNcIiwgXCJjbGFzc05hbWVcIiwgXCJjaGlsZHJlblwiLCBcInRhZ05hbWVcIl0pO1xuXG4gIHZhciBjbGFzc1N0cmluZyA9ICgwLCBfY2xhc3NuYW1lc1tcImRlZmF1bHRcIl0pKHByZWZpeENscywgY2xhc3NOYW1lKTtcbiAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KHRhZ05hbWUsICgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7XG4gICAgY2xhc3NOYW1lOiBjbGFzc1N0cmluZ1xuICB9LCBvdGhlcnMpLCBjaGlsZHJlbik7XG59O1xuXG52YXIgQmFzaWNMYXlvdXQgPSBmdW5jdGlvbiBCYXNpY0xheW91dChwcm9wcykge1xuICB2YXIgX2NsYXNzTmFtZXM7XG5cbiAgdmFyIF9SZWFjdCR1c2VDb250ZXh0MiA9IFJlYWN0LnVzZUNvbnRleHQoX2NvbmZpZ1Byb3ZpZGVyLkNvbmZpZ0NvbnRleHQpLFxuICAgICAgZGlyZWN0aW9uID0gX1JlYWN0JHVzZUNvbnRleHQyLmRpcmVjdGlvbjtcblxuICB2YXIgX1JlYWN0JHVzZVN0YXRlID0gUmVhY3QudXNlU3RhdGUoW10pLFxuICAgICAgX1JlYWN0JHVzZVN0YXRlMiA9ICgwLCBfc2xpY2VkVG9BcnJheTJbXCJkZWZhdWx0XCJdKShfUmVhY3QkdXNlU3RhdGUsIDIpLFxuICAgICAgc2lkZXJzID0gX1JlYWN0JHVzZVN0YXRlMlswXSxcbiAgICAgIHNldFNpZGVycyA9IF9SZWFjdCR1c2VTdGF0ZTJbMV07XG5cbiAgdmFyIHByZWZpeENscyA9IHByb3BzLnByZWZpeENscyxcbiAgICAgIGNsYXNzTmFtZSA9IHByb3BzLmNsYXNzTmFtZSxcbiAgICAgIGNoaWxkcmVuID0gcHJvcHMuY2hpbGRyZW4sXG4gICAgICBoYXNTaWRlciA9IHByb3BzLmhhc1NpZGVyLFxuICAgICAgVGFnID0gcHJvcHMudGFnTmFtZSxcbiAgICAgIG90aGVycyA9IF9fcmVzdChwcm9wcywgW1wicHJlZml4Q2xzXCIsIFwiY2xhc3NOYW1lXCIsIFwiY2hpbGRyZW5cIiwgXCJoYXNTaWRlclwiLCBcInRhZ05hbWVcIl0pO1xuXG4gIHZhciBjbGFzc1N0cmluZyA9ICgwLCBfY2xhc3NuYW1lc1tcImRlZmF1bHRcIl0pKHByZWZpeENscywgKF9jbGFzc05hbWVzID0ge30sICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItaGFzLXNpZGVyXCIpLCB0eXBlb2YgaGFzU2lkZXIgPT09ICdib29sZWFuJyA/IGhhc1NpZGVyIDogc2lkZXJzLmxlbmd0aCA+IDApLCAoMCwgX2RlZmluZVByb3BlcnR5MltcImRlZmF1bHRcIl0pKF9jbGFzc05hbWVzLCBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLXJ0bFwiKSwgZGlyZWN0aW9uID09PSAncnRsJyksIF9jbGFzc05hbWVzKSwgY2xhc3NOYW1lKTtcbiAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KExheW91dENvbnRleHQuUHJvdmlkZXIsIHtcbiAgICB2YWx1ZToge1xuICAgICAgc2lkZXJIb29rOiB7XG4gICAgICAgIGFkZFNpZGVyOiBmdW5jdGlvbiBhZGRTaWRlcihpZCkge1xuICAgICAgICAgIHNldFNpZGVycyhmdW5jdGlvbiAocHJldikge1xuICAgICAgICAgICAgcmV0dXJuIFtdLmNvbmNhdCgoMCwgX3RvQ29uc3VtYWJsZUFycmF5MltcImRlZmF1bHRcIl0pKHByZXYpLCBbaWRdKTtcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSxcbiAgICAgICAgcmVtb3ZlU2lkZXI6IGZ1bmN0aW9uIHJlbW92ZVNpZGVyKGlkKSB7XG4gICAgICAgICAgc2V0U2lkZXJzKGZ1bmN0aW9uIChwcmV2KSB7XG4gICAgICAgICAgICByZXR1cm4gcHJldi5maWx0ZXIoZnVuY3Rpb24gKGN1cnJlbnRJZCkge1xuICAgICAgICAgICAgICByZXR1cm4gY3VycmVudElkICE9PSBpZDtcbiAgICAgICAgICAgIH0pO1xuICAgICAgICAgIH0pO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9LCAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChUYWcsICgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7XG4gICAgY2xhc3NOYW1lOiBjbGFzc1N0cmluZ1xuICB9LCBvdGhlcnMpLCBjaGlsZHJlbikpO1xufTtcblxudmFyIExheW91dCA9IGdlbmVyYXRvcih7XG4gIHN1ZmZpeENsczogJ2xheW91dCcsXG4gIHRhZ05hbWU6ICdzZWN0aW9uJyxcbiAgZGlzcGxheU5hbWU6ICdMYXlvdXQnXG59KShCYXNpY0xheW91dCk7XG52YXIgSGVhZGVyID0gZ2VuZXJhdG9yKHtcbiAgc3VmZml4Q2xzOiAnbGF5b3V0LWhlYWRlcicsXG4gIHRhZ05hbWU6ICdoZWFkZXInLFxuICBkaXNwbGF5TmFtZTogJ0hlYWRlcidcbn0pKEJhc2ljKTtcbmV4cG9ydHMuSGVhZGVyID0gSGVhZGVyO1xudmFyIEZvb3RlciA9IGdlbmVyYXRvcih7XG4gIHN1ZmZpeENsczogJ2xheW91dC1mb290ZXInLFxuICB0YWdOYW1lOiAnZm9vdGVyJyxcbiAgZGlzcGxheU5hbWU6ICdGb290ZXInXG59KShCYXNpYyk7XG5leHBvcnRzLkZvb3RlciA9IEZvb3RlcjtcbnZhciBDb250ZW50ID0gZ2VuZXJhdG9yKHtcbiAgc3VmZml4Q2xzOiAnbGF5b3V0LWNvbnRlbnQnLFxuICB0YWdOYW1lOiAnbWFpbicsXG4gIGRpc3BsYXlOYW1lOiAnQ29udGVudCdcbn0pKEJhc2ljKTtcbmV4cG9ydHMuQ29udGVudCA9IENvbnRlbnQ7XG52YXIgX2RlZmF1bHQgPSBMYXlvdXQ7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IF9kZWZhdWx0OyIsImltcG9ydCBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IE5leHRQYWdlIH0gZnJvbSAnbmV4dCc7XHJcbmltcG9ydCBIZWFkIGZyb20gJ25leHQvaGVhZCc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgQ29sLCBUb29sdGlwLCBMYXlvdXQgfSBmcm9tICdhbnRkJztcclxuXHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkSGVhZGVyLFxyXG4gIFN0eWxlZExheW91dCxcclxuICBTdHlsZWREaXYsXHJcbiAgU3R5bGVkTG9nb1dyYXBwZXIsXHJcbiAgU3R5bGVkTG9nbyxcclxuICBTdHlsZWRMb2dvRGl2LFxyXG59IGZyb20gJy4uL3N0eWxlcy9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgRm9sZGVyUGF0aFF1ZXJ5LCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBiYWNrVG9NYWluUGFnZSB9IGZyb20gJy4uL3V0aWxzL3BhZ2VzJztcclxuaW1wb3J0IHsgSGVhZGVyIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2hlYWRlcic7XHJcbmltcG9ydCB7IENvbnRlbnRTd2l0Y2hpbmcgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvY29udGVudC9jb25zdGVudF9zd2l0Y2hpbmcnO1xyXG5pbXBvcnQgeyBNb2Rlc1NlbGVjdGlvbiB9IGZyb20gJy4uL2NvbXBvbmVudHMvbW9kZXMvbW9kZXNTZWxlY3Rpb24nO1xyXG5pbXBvcnQgeyBGb290ZXIgfSBmcm9tICdhbnRkL2xpYi9sYXlvdXQvbGF5b3V0JztcclxuXHJcblxyXG5jb25zdCBJbmRleDogTmV4dFBhZ2U8Rm9sZGVyUGF0aFF1ZXJ5PiA9ICgpID0+IHtcclxuICAvLyBXZSBncmFiIHRoZSBxdWVyeSBmcm9tIHRoZSBVUkw6XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgY29uc3QgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQgPVxyXG4gICAgISFxdWVyeS5ydW5fbnVtYmVyICYmICEhcXVlcnkuZGF0YXNldF9uYW1lO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFN0eWxlZERpdj5cclxuICAgICAgPEhlYWQ+XHJcbiAgICAgICAgPHNjcmlwdFxyXG4gICAgICAgICAgY3Jvc3NPcmlnaW49XCJhbm9ueW1vdXNcIlxyXG4gICAgICAgICAgdHlwZT1cInRleHQvamF2YXNjcmlwdFwiXHJcbiAgICAgICAgICBzcmM9XCIuL2pzcm9vdC01LjguMC9zY3JpcHRzL0pTUm9vdENvcmUuanM/MmQmaGlzdCZtb3JlMmRcIlxyXG4gICAgICAgID48L3NjcmlwdD5cclxuICAgICAgPC9IZWFkPlxyXG4gICAgICA8U3R5bGVkTGF5b3V0PlxyXG4gICAgICAgIDxTdHlsZWRIZWFkZXI+XHJcbiAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICA8Q29sIHN0eWxlPXt7IGRpc3BsYXk6ICdmbGV4JywgYWxpZ25JdGVtczogJ2NlbnRlcicgfX0+XHJcbiAgICAgICAgICAgICAgPFRvb2x0aXAgdGl0bGU9XCJCYWNrIHRvIG1haW4gcGFnZVwiIHBsYWNlbWVudD1cImJvdHRvbUxlZnRcIj5cclxuICAgICAgICAgICAgICAgIDxTdHlsZWRMb2dvRGl2PlxyXG4gICAgICAgICAgICAgICAgICA8U3R5bGVkTG9nb1dyYXBwZXIgb25DbGljaz17KGUpID0+IGJhY2tUb01haW5QYWdlKGUpfT5cclxuICAgICAgICAgICAgICAgICAgICA8U3R5bGVkTG9nbyBzcmM9XCIuL2ltYWdlcy9DTVNsb2dvX3doaXRlX3JlZF9ub2xhYmVsXzEwMjRfTWF5MjAxNC5wbmdcIiAvPlxyXG4gICAgICAgICAgICAgICAgICA8L1N0eWxlZExvZ29XcmFwcGVyPlxyXG4gICAgICAgICAgICAgICAgPC9TdHlsZWRMb2dvRGl2PlxyXG4gICAgICAgICAgICAgIDwvVG9vbHRpcD5cclxuICAgICAgICAgICAgIDxNb2Rlc1NlbGVjdGlvbiAvPlxyXG4gICAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgPEhlYWRlclxyXG4gICAgICAgICAgICBpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZD17aXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWR9XHJcbiAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9TdHlsZWRIZWFkZXI+XHJcbiAgICAgICAgPENvbnRlbnRTd2l0Y2hpbmcgLz5cclxuICAgICAgICA8Rm9vdGVyIC8+XHJcbiAgICAgIDwvU3R5bGVkTGF5b3V0PlxyXG4gICAgPC9TdHlsZWREaXY+XHJcbiAgKTtcclxufTtcclxuXHJcbmV4cG9ydCBkZWZhdWx0IEluZGV4O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9