webpackHotUpdate_N_E("pages/index",{

/***/ "./components/navigation/freeSearchResultModal.tsx":
/*!*********************************************************!*\
  !*** ./components/navigation/freeSearchResultModal.tsx ***!
  \*********************************************************/
/*! exports provided: SearchModal */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "SearchModal", function() { return SearchModal; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! qs */ "./node_modules/qs/lib/index.js");
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(qs__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var _containers_search_SearchResults__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../containers/search/SearchResults */ "./containers/search/SearchResults.tsx");
/* harmony import */ var _hooks_useSearch__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../hooks/useSearch */ "./hooks/useSearch.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _selectedData__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ./selectedData */ "./components/navigation/selectedData.tsx");
/* harmony import */ var _Nav__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../Nav */ "./components/Nav.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_15__ = __webpack_require__(/*! antd/lib/modal/Modal */ "./node_modules/antd/lib/modal/Modal.js");
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_15___default = /*#__PURE__*/__webpack_require__.n(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_15__);




var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/navigation/freeSearchResultModal.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3___default.a.createElement;














var open_a_new_tab = function open_a_new_tab(query) {
  window.open(query, '_blank');
};

var SearchModal = function SearchModal(_ref) {
  _s();

  var setModalState = _ref.setModalState,
      modalState = _ref.modalState,
      search_run_number = _ref.search_run_number,
      search_dataset_name = _ref.search_dataset_name,
      setSearchDatasetName = _ref.setSearchDatasetName,
      setSearchRunNumber = _ref.setSearchRunNumber;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_5__["useRouter"])();
  var query = router.query;
  var dataset = query.dataset_name ? query.dataset_name : '';

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(dataset),
      datasetName = _useState[0],
      setDatasetName = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(false),
      openRunInNewTab = _useState2[0],
      toggleRunInNewTab = _useState2[1];

  var run = query.run_number ? query.run_number : '';

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_3__["useState"])(run),
      runNumber = _useState3[0],
      setRunNumber = _useState3[1];

  Object(react__WEBPACK_IMPORTED_MODULE_3__["useEffect"])(function () {
    var run = query.run_number ? query.run_number : '';
    var dataset = query.dataset_name ? query.dataset_name : '';
    setDatasetName(dataset);
    setRunNumber(run);
  }, [query.dataset_name, query.run_number]);

  var onClosing = function onClosing() {
    setModalState(false);
  };

  var searchHandler = function searchHandler(run_number, dataset_name) {
    setDatasetName(dataset_name);
    setRunNumber(run_number);
  };

  var navigationHandler = function navigationHandler(search_by_run_number, search_by_dataset_name) {
    setSearchRunNumber(search_by_run_number);
    setSearchDatasetName(search_by_dataset_name);
  };

  var _useSearch = Object(_hooks_useSearch__WEBPACK_IMPORTED_MODULE_9__["useSearch"])(search_run_number, search_dataset_name),
      results_grouped = _useSearch.results_grouped,
      searching = _useSearch.searching,
      isLoading = _useSearch.isLoading,
      errors = _useSearch.errors;

  var onOk = /*#__PURE__*/function () {
    var _ref2 = Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default.a.mark(function _callee() {
      var params, new_tab_query_params, current_root;
      return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default.a.wrap(function _callee$(_context) {
        while (1) {
          switch (_context.prev = _context.next) {
            case 0:
              if (!openRunInNewTab) {
                _context.next = 7;
                break;
              }

              params = form.getFieldsValue();
              new_tab_query_params = qs__WEBPACK_IMPORTED_MODULE_4___default.a.stringify(Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_14__["getChangedQueryParams"])(params, query)); //root url is ends with first '?'. I can't use just root url from config.config, because
              //in dev env it use localhost:8081/dqm/dev (this is old backend url from where I'm getting data),
              //but I need localhost:3000

              current_root = window.location.href.split('/?')[0];
              open_a_new_tab("".concat(current_root, "/?").concat(new_tab_query_params));
              _context.next = 9;
              break;

            case 7:
              _context.next = 9;
              return form.submit();

            case 9:
              onClosing();

            case 10:
            case "end":
              return _context.stop();
          }
        }
      }, _callee);
    }));

    return function onOk() {
      return _ref2.apply(this, arguments);
    };
  }();

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_6__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_15___default.a, {
    title: "Search data",
    visible: modalState,
    onCancel: function onCancel() {
      return onClosing();
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_10__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_11__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return onClosing();
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 105,
        columnNumber: 9
      }
    }, "Close"), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_10__["StyledButton"], {
      key: "OK",
      onClick: onOk,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 113,
        columnNumber: 9
      }
    }, "OK")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 100,
      columnNumber: 5
    }
  }, modalState && __jsx(react__WEBPACK_IMPORTED_MODULE_3___default.a.Fragment, null, __jsx(_Nav__WEBPACK_IMPORTED_MODULE_13__["default"], {
    initial_search_run_number: search_run_number,
    initial_search_dataset_name: search_dataset_name,
    defaultDatasetName: datasetName,
    defaultRunNumber: runNumber,
    handler: navigationHandler,
    type: "top",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 120,
      columnNumber: 11
    }
  }), __jsx(_selectedData__WEBPACK_IMPORTED_MODULE_12__["SelectedData"], {
    form: form,
    dataset_name: datasetName,
    run_number: runNumber,
    toggleRunInNewTab: toggleRunInNewTab,
    openRunInNewTab: openRunInNewTab,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 128,
      columnNumber: 11
    }
  }), searching ? __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 136,
      columnNumber: 13
    }
  }, __jsx(_containers_search_SearchResults__WEBPACK_IMPORTED_MODULE_8__["default"], {
    handler: searchHandler,
    isLoading: isLoading,
    results_grouped: results_grouped,
    errors: errors,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 137,
      columnNumber: 15
    }
  })) : __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["ResultsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 145,
      columnNumber: 13
    }
  })));
};

_s(SearchModal, "cJSZLTqxYxam8F0Rr2yyVtEoUY8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_5__["useRouter"], _hooks_useSearch__WEBPACK_IMPORTED_MODULE_9__["useSearch"], antd__WEBPACK_IMPORTED_MODULE_6__["Form"].useForm];
});

_c = SearchModal;

var _c;

$RefreshReg$(_c, "SearchModal");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ }),

/***/ "./node_modules/antd/lib/_util/hooks/usePatchElement.js":
/*!**************************************************************!*\
  !*** ./node_modules/antd/lib/_util/hooks/usePatchElement.js ***!
  \**************************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = usePatchElement;

var _toConsumableArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/toConsumableArray */ "./node_modules/@babel/runtime/helpers/toConsumableArray.js"));

var _slicedToArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/slicedToArray */ "./node_modules/@babel/runtime/helpers/slicedToArray.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

function usePatchElement() {
  var _React$useState = React.useState([]),
      _React$useState2 = (0, _slicedToArray2["default"])(_React$useState, 2),
      elements = _React$useState2[0],
      setElements = _React$useState2[1];

  var patchElement = React.useCallback(function (element) {
    // append a new element to elements (and create a new ref)
    setElements(function (originElements) {
      return [].concat((0, _toConsumableArray2["default"])(originElements), [element]);
    }); // return a function that removes the new element out of elements (and create a new ref)
    // it works a little like useEffect

    return function () {
      setElements(function (originElements) {
        return originElements.filter(function (ele) {
          return ele !== element;
        });
      });
    };
  }, []);
  return [elements, patchElement];
}

/***/ }),

/***/ "./node_modules/antd/lib/_util/raf.js":
/*!********************************************!*\
  !*** ./node_modules/antd/lib/_util/raf.js ***!
  \********************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = wrapperRaf;

var _raf = _interopRequireDefault(__webpack_require__(/*! rc-util/lib/raf */ "./node_modules/rc-util/lib/raf.js"));

var id = 0;
var ids = {}; // Support call raf with delay specified frame

function wrapperRaf(callback) {
  var delayFrames = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 1;
  var myId = id++;
  var restFrames = delayFrames;

  function internalCallback() {
    restFrames -= 1;

    if (restFrames <= 0) {
      callback();
      delete ids[myId];
    } else {
      ids[myId] = (0, _raf["default"])(internalCallback);
    }
  }

  ids[myId] = (0, _raf["default"])(internalCallback);
  return myId;
}

wrapperRaf.cancel = function cancel(pid) {
  if (pid === undefined) return;

  _raf["default"].cancel(ids[pid]);

  delete ids[pid];
};

wrapperRaf.ids = ids; // export this for test usage

/***/ }),

/***/ "./node_modules/antd/lib/_util/unreachableException.js":
/*!*************************************************************!*\
  !*** ./node_modules/antd/lib/_util/unreachableException.js ***!
  \*************************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _classCallCheck2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/classCallCheck */ "./node_modules/@babel/runtime/helpers/classCallCheck.js"));

var UnreachableException = function UnreachableException(value) {
  (0, _classCallCheck2["default"])(this, UnreachableException);
  return new Error("unreachable case: ".concat(JSON.stringify(value)));
};

exports["default"] = UnreachableException;

/***/ }),

/***/ "./node_modules/antd/lib/_util/wave.js":
/*!*********************************************!*\
  !*** ./node_modules/antd/lib/_util/wave.js ***!
  \*********************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _classCallCheck2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/classCallCheck */ "./node_modules/@babel/runtime/helpers/classCallCheck.js"));

var _createClass2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/createClass */ "./node_modules/@babel/runtime/helpers/createClass.js"));

var _assertThisInitialized2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/assertThisInitialized */ "./node_modules/@babel/runtime/helpers/assertThisInitialized.js"));

var _inherits2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/inherits */ "./node_modules/@babel/runtime/helpers/inherits.js"));

var _createSuper2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/createSuper */ "./node_modules/@babel/runtime/helpers/createSuper.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _ref2 = __webpack_require__(/*! rc-util/lib/ref */ "./node_modules/rc-util/lib/ref.js");

var _raf = _interopRequireDefault(__webpack_require__(/*! ./raf */ "./node_modules/antd/lib/_util/raf.js"));

var _configProvider = __webpack_require__(/*! ../config-provider */ "./node_modules/antd/lib/config-provider/index.js");

var _reactNode = __webpack_require__(/*! ./reactNode */ "./node_modules/antd/lib/_util/reactNode.js");

var styleForPseudo; // Where el is the DOM element you'd like to test for visibility

function isHidden(element) {
  if (false) {}

  return !element || element.offsetParent === null || element.hidden;
}

function isNotGrey(color) {
  // eslint-disable-next-line no-useless-escape
  var match = (color || '').match(/rgba?\((\d*), (\d*), (\d*)(, [\d.]*)?\)/);

  if (match && match[1] && match[2] && match[3]) {
    return !(match[1] === match[2] && match[2] === match[3]);
  }

  return true;
}

var Wave = /*#__PURE__*/function (_React$Component) {
  (0, _inherits2["default"])(Wave, _React$Component);

  var _super = (0, _createSuper2["default"])(Wave);

  function Wave() {
    var _this;

    (0, _classCallCheck2["default"])(this, Wave);
    _this = _super.apply(this, arguments);
    _this.containerRef = /*#__PURE__*/React.createRef();
    _this.animationStart = false;
    _this.destroyed = false;

    _this.onClick = function (node, waveColor) {
      if (!node || isHidden(node) || node.className.indexOf('-leave') >= 0) {
        return;
      }

      var insertExtraNode = _this.props.insertExtraNode;
      _this.extraNode = document.createElement('div');

      var _assertThisInitialize = (0, _assertThisInitialized2["default"])(_this),
          extraNode = _assertThisInitialize.extraNode;

      var getPrefixCls = _this.context.getPrefixCls;
      extraNode.className = "".concat(getPrefixCls(''), "-click-animating-node");

      var attributeName = _this.getAttributeName();

      node.setAttribute(attributeName, 'true'); // Not white or transparent or grey

      styleForPseudo = styleForPseudo || document.createElement('style');

      if (waveColor && waveColor !== '#ffffff' && waveColor !== 'rgb(255, 255, 255)' && isNotGrey(waveColor) && !/rgba\((?:\d*, ){3}0\)/.test(waveColor) && // any transparent rgba color
      waveColor !== 'transparent') {
        // Add nonce if CSP exist
        if (_this.csp && _this.csp.nonce) {
          styleForPseudo.nonce = _this.csp.nonce;
        }

        extraNode.style.borderColor = waveColor;
        styleForPseudo.innerHTML = "\n      [".concat(getPrefixCls(''), "-click-animating-without-extra-node='true']::after, .").concat(getPrefixCls(''), "-click-animating-node {\n        --antd-wave-shadow-color: ").concat(waveColor, ";\n      }");

        if (!node.ownerDocument.body.contains(styleForPseudo)) {
          node.ownerDocument.body.appendChild(styleForPseudo);
        }
      }

      if (insertExtraNode) {
        node.appendChild(extraNode);
      }

      ['transition', 'animation'].forEach(function (name) {
        node.addEventListener("".concat(name, "start"), _this.onTransitionStart);
        node.addEventListener("".concat(name, "end"), _this.onTransitionEnd);
      });
    };

    _this.onTransitionStart = function (e) {
      if (_this.destroyed) {
        return;
      }

      var node = _this.containerRef.current;

      if (!e || e.target !== node || _this.animationStart) {
        return;
      }

      _this.resetEffect(node);
    };

    _this.onTransitionEnd = function (e) {
      if (!e || e.animationName !== 'fadeEffect') {
        return;
      }

      _this.resetEffect(e.target);
    };

    _this.bindAnimationEvent = function (node) {
      if (!node || !node.getAttribute || node.getAttribute('disabled') || node.className.indexOf('disabled') >= 0) {
        return;
      }

      var onClick = function onClick(e) {
        // Fix radio button click twice
        if (e.target.tagName === 'INPUT' || isHidden(e.target)) {
          return;
        }

        _this.resetEffect(node); // Get wave color from target


        var waveColor = getComputedStyle(node).getPropertyValue('border-top-color') || // Firefox Compatible
        getComputedStyle(node).getPropertyValue('border-color') || getComputedStyle(node).getPropertyValue('background-color');
        _this.clickWaveTimeoutId = window.setTimeout(function () {
          return _this.onClick(node, waveColor);
        }, 0);

        _raf["default"].cancel(_this.animationStartId);

        _this.animationStart = true; // Render to trigger transition event cost 3 frames. Let's delay 10 frames to reset this.

        _this.animationStartId = (0, _raf["default"])(function () {
          _this.animationStart = false;
        }, 10);
      };

      node.addEventListener('click', onClick, true);
      return {
        cancel: function cancel() {
          node.removeEventListener('click', onClick, true);
        }
      };
    };

    _this.renderWave = function (_ref) {
      var csp = _ref.csp;
      var children = _this.props.children;
      _this.csp = csp;
      if (! /*#__PURE__*/React.isValidElement(children)) return children;
      var ref = _this.containerRef;

      if ((0, _ref2.supportRef)(children)) {
        ref = (0, _ref2.composeRef)(children.ref, _this.containerRef);
      }

      return (0, _reactNode.cloneElement)(children, {
        ref: ref
      });
    };

    return _this;
  }

  (0, _createClass2["default"])(Wave, [{
    key: "componentDidMount",
    value: function componentDidMount() {
      var node = this.containerRef.current;

      if (!node || node.nodeType !== 1) {
        return;
      }

      this.instance = this.bindAnimationEvent(node);
    }
  }, {
    key: "componentWillUnmount",
    value: function componentWillUnmount() {
      if (this.instance) {
        this.instance.cancel();
      }

      if (this.clickWaveTimeoutId) {
        clearTimeout(this.clickWaveTimeoutId);
      }

      this.destroyed = true;
    }
  }, {
    key: "getAttributeName",
    value: function getAttributeName() {
      var getPrefixCls = this.context.getPrefixCls;
      var insertExtraNode = this.props.insertExtraNode;
      return insertExtraNode ? "".concat(getPrefixCls(''), "-click-animating") : "".concat(getPrefixCls(''), "-click-animating-without-extra-node");
    }
  }, {
    key: "resetEffect",
    value: function resetEffect(node) {
      var _this2 = this;

      if (!node || node === this.extraNode || !(node instanceof Element)) {
        return;
      }

      var insertExtraNode = this.props.insertExtraNode;
      var attributeName = this.getAttributeName();
      node.setAttribute(attributeName, 'false'); // edge has bug on `removeAttribute` #14466

      if (styleForPseudo) {
        styleForPseudo.innerHTML = '';
      }

      if (insertExtraNode && this.extraNode && node.contains(this.extraNode)) {
        node.removeChild(this.extraNode);
      }

      ['transition', 'animation'].forEach(function (name) {
        node.removeEventListener("".concat(name, "start"), _this2.onTransitionStart);
        node.removeEventListener("".concat(name, "end"), _this2.onTransitionEnd);
      });
    }
  }, {
    key: "render",
    value: function render() {
      return /*#__PURE__*/React.createElement(_configProvider.ConfigConsumer, null, this.renderWave);
    }
  }]);
  return Wave;
}(React.Component);

exports["default"] = Wave;
Wave.contextType = _configProvider.ConfigContext;

/***/ }),

/***/ "./node_modules/antd/lib/button/LoadingIcon.js":
/*!*****************************************************!*\
  !*** ./node_modules/antd/lib/button/LoadingIcon.js ***!
  \*****************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _react = _interopRequireDefault(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _rcMotion = _interopRequireDefault(__webpack_require__(/*! rc-motion */ "./node_modules/rc-motion/es/index.js"));

var _LoadingOutlined = _interopRequireDefault(__webpack_require__(/*! @ant-design/icons/LoadingOutlined */ "./node_modules/@ant-design/icons/LoadingOutlined.js"));

var getCollapsedWidth = function getCollapsedWidth() {
  return {
    width: 0,
    opacity: 0,
    transform: 'scale(0)'
  };
};

var getRealWidth = function getRealWidth(node) {
  return {
    width: node.scrollWidth,
    opacity: 1,
    transform: 'scale(1)'
  };
};

var LoadingIcon = function LoadingIcon(_ref) {
  var prefixCls = _ref.prefixCls,
      loading = _ref.loading,
      existIcon = _ref.existIcon;
  var visible = !!loading;

  if (existIcon) {
    return /*#__PURE__*/_react["default"].createElement("span", {
      className: "".concat(prefixCls, "-loading-icon")
    }, /*#__PURE__*/_react["default"].createElement(_LoadingOutlined["default"], null));
  }

  return /*#__PURE__*/_react["default"].createElement(_rcMotion["default"], {
    visible: visible // We do not really use this motionName
    ,
    motionName: "".concat(prefixCls, "-loading-icon-motion"),
    removeOnLeave: true,
    onAppearStart: getCollapsedWidth,
    onAppearActive: getRealWidth,
    onEnterStart: getCollapsedWidth,
    onEnterActive: getRealWidth,
    onLeaveStart: getRealWidth,
    onLeaveActive: getCollapsedWidth
  }, function (_ref2, ref) {
    var className = _ref2.className,
        style = _ref2.style;
    return /*#__PURE__*/_react["default"].createElement("span", {
      className: "".concat(prefixCls, "-loading-icon"),
      style: style,
      ref: ref
    }, /*#__PURE__*/_react["default"].createElement(_LoadingOutlined["default"], {
      className: className
    }));
  });
};

var _default = LoadingIcon;
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/button/button-group.js":
/*!******************************************************!*\
  !*** ./node_modules/antd/lib/button/button-group.js ***!
  \******************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var _defineProperty2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/defineProperty */ "./node_modules/@babel/runtime/helpers/defineProperty.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _classnames = _interopRequireDefault(__webpack_require__(/*! classnames */ "./node_modules/classnames/index.js"));

var _configProvider = __webpack_require__(/*! ../config-provider */ "./node_modules/antd/lib/config-provider/index.js");

var _unreachableException = _interopRequireDefault(__webpack_require__(/*! ../_util/unreachableException */ "./node_modules/antd/lib/_util/unreachableException.js"));

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

var ButtonGroup = function ButtonGroup(props) {
  return /*#__PURE__*/React.createElement(_configProvider.ConfigConsumer, null, function (_ref) {
    var _classNames;

    var getPrefixCls = _ref.getPrefixCls,
        direction = _ref.direction;

    var customizePrefixCls = props.prefixCls,
        size = props.size,
        className = props.className,
        others = __rest(props, ["prefixCls", "size", "className"]);

    var prefixCls = getPrefixCls('btn-group', customizePrefixCls); // large => lg
    // small => sm

    var sizeCls = '';

    switch (size) {
      case 'large':
        sizeCls = 'lg';
        break;

      case 'small':
        sizeCls = 'sm';
        break;

      case 'middle':
      case undefined:
        break;

      default:
        // eslint-disable-next-line no-console
        console.warn(new _unreachableException["default"](size));
    }

    var classes = (0, _classnames["default"])(prefixCls, (_classNames = {}, (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-").concat(sizeCls), sizeCls), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-rtl"), direction === 'rtl'), _classNames), className);
    return /*#__PURE__*/React.createElement("div", (0, _extends2["default"])({}, others, {
      className: classes
    }));
  });
};

var _default = ButtonGroup;
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/button/button.js":
/*!************************************************!*\
  !*** ./node_modules/antd/lib/button/button.js ***!
  \************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.convertLegacyProps = convertLegacyProps;
exports["default"] = void 0;

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var _defineProperty2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/defineProperty */ "./node_modules/@babel/runtime/helpers/defineProperty.js"));

var _slicedToArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/slicedToArray */ "./node_modules/@babel/runtime/helpers/slicedToArray.js"));

var _typeof2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/typeof */ "./node_modules/@babel/runtime/helpers/typeof.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _classnames = _interopRequireDefault(__webpack_require__(/*! classnames */ "./node_modules/classnames/index.js"));

var _omit = _interopRequireDefault(__webpack_require__(/*! rc-util/lib/omit */ "./node_modules/rc-util/lib/omit.js"));

var _buttonGroup = _interopRequireDefault(__webpack_require__(/*! ./button-group */ "./node_modules/antd/lib/button/button-group.js"));

var _configProvider = __webpack_require__(/*! ../config-provider */ "./node_modules/antd/lib/config-provider/index.js");

var _wave = _interopRequireDefault(__webpack_require__(/*! ../_util/wave */ "./node_modules/antd/lib/_util/wave.js"));

var _type = __webpack_require__(/*! ../_util/type */ "./node_modules/antd/lib/_util/type.js");

var _devWarning = _interopRequireDefault(__webpack_require__(/*! ../_util/devWarning */ "./node_modules/antd/lib/_util/devWarning.js"));

var _SizeContext = _interopRequireDefault(__webpack_require__(/*! ../config-provider/SizeContext */ "./node_modules/antd/lib/config-provider/SizeContext.js"));

var _LoadingIcon = _interopRequireDefault(__webpack_require__(/*! ./LoadingIcon */ "./node_modules/antd/lib/button/LoadingIcon.js"));

var _reactNode = __webpack_require__(/*! ../_util/reactNode */ "./node_modules/antd/lib/_util/reactNode.js");

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
/* eslint-disable react/button-has-type */


var rxTwoCNChar = /^[\u4e00-\u9fa5]{2}$/;
var isTwoCNChar = rxTwoCNChar.test.bind(rxTwoCNChar);

function isString(str) {
  return typeof str === 'string';
}

function isUnborderedButtonType(type) {
  return type === 'text' || type === 'link';
} // Insert one space between two chinese characters automatically.


function insertSpace(child, needInserted) {
  // Check the child if is undefined or null.
  if (child == null) {
    return;
  }

  var SPACE = needInserted ? ' ' : ''; // strictNullChecks oops.

  if (typeof child !== 'string' && typeof child !== 'number' && isString(child.type) && isTwoCNChar(child.props.children)) {
    return (0, _reactNode.cloneElement)(child, {
      children: child.props.children.split('').join(SPACE)
    });
  }

  if (typeof child === 'string') {
    if (isTwoCNChar(child)) {
      child = child.split('').join(SPACE);
    }

    return /*#__PURE__*/React.createElement("span", null, child);
  }

  return child;
}

function spaceChildren(children, needInserted) {
  var isPrevChildPure = false;
  var childList = [];
  React.Children.forEach(children, function (child) {
    var type = (0, _typeof2["default"])(child);
    var isCurrentChildPure = type === 'string' || type === 'number';

    if (isPrevChildPure && isCurrentChildPure) {
      var lastIndex = childList.length - 1;
      var lastChild = childList[lastIndex];
      childList[lastIndex] = "".concat(lastChild).concat(child);
    } else {
      childList.push(child);
    }

    isPrevChildPure = isCurrentChildPure;
  }); // Pass to React.Children.map to auto fill key

  return React.Children.map(childList, function (child) {
    return insertSpace(child, needInserted);
  });
}

var ButtonTypes = (0, _type.tuple)('default', 'primary', 'ghost', 'dashed', 'link', 'text');
var ButtonShapes = (0, _type.tuple)('circle', 'round');
var ButtonHTMLTypes = (0, _type.tuple)('submit', 'button', 'reset');

function convertLegacyProps(type) {
  if (type === 'danger') {
    return {
      danger: true
    };
  }

  return {
    type: type
  };
}

var InternalButton = function InternalButton(props, ref) {
  var _classNames;

  var _props$loading = props.loading,
      loading = _props$loading === void 0 ? false : _props$loading,
      customizePrefixCls = props.prefixCls,
      type = props.type,
      danger = props.danger,
      shape = props.shape,
      customizeSize = props.size,
      className = props.className,
      children = props.children,
      icon = props.icon,
      _props$ghost = props.ghost,
      ghost = _props$ghost === void 0 ? false : _props$ghost,
      _props$block = props.block,
      block = _props$block === void 0 ? false : _props$block,
      _props$htmlType = props.htmlType,
      htmlType = _props$htmlType === void 0 ? 'button' : _props$htmlType,
      rest = __rest(props, ["loading", "prefixCls", "type", "danger", "shape", "size", "className", "children", "icon", "ghost", "block", "htmlType"]);

  var size = React.useContext(_SizeContext["default"]);

  var _React$useState = React.useState(!!loading),
      _React$useState2 = (0, _slicedToArray2["default"])(_React$useState, 2),
      innerLoading = _React$useState2[0],
      setLoading = _React$useState2[1];

  var _React$useState3 = React.useState(false),
      _React$useState4 = (0, _slicedToArray2["default"])(_React$useState3, 2),
      hasTwoCNChar = _React$useState4[0],
      setHasTwoCNChar = _React$useState4[1];

  var _React$useContext = React.useContext(_configProvider.ConfigContext),
      getPrefixCls = _React$useContext.getPrefixCls,
      autoInsertSpaceInButton = _React$useContext.autoInsertSpaceInButton,
      direction = _React$useContext.direction;

  var buttonRef = ref || /*#__PURE__*/React.createRef();
  var delayTimeoutRef = React.useRef();

  var isNeedInserted = function isNeedInserted() {
    return React.Children.count(children) === 1 && !icon && !isUnborderedButtonType(type);
  };

  var fixTwoCNChar = function fixTwoCNChar() {
    // Fix for HOC usage like <FormatMessage />
    if (!buttonRef || !buttonRef.current || autoInsertSpaceInButton === false) {
      return;
    }

    var buttonText = buttonRef.current.textContent;

    if (isNeedInserted() && isTwoCNChar(buttonText)) {
      if (!hasTwoCNChar) {
        setHasTwoCNChar(true);
      }
    } else if (hasTwoCNChar) {
      setHasTwoCNChar(false);
    }
  }; // =============== Update Loading ===============


  var loadingOrDelay;

  if ((0, _typeof2["default"])(loading) === 'object' && loading.delay) {
    loadingOrDelay = loading.delay || true;
  } else {
    loadingOrDelay = !!loading;
  }

  React.useEffect(function () {
    clearTimeout(delayTimeoutRef.current);

    if (typeof loadingOrDelay === 'number') {
      delayTimeoutRef.current = window.setTimeout(function () {
        setLoading(loadingOrDelay);
      }, loadingOrDelay);
    } else {
      setLoading(loadingOrDelay);
    }
  }, [loadingOrDelay]);
  React.useEffect(fixTwoCNChar, [buttonRef]);

  var handleClick = function handleClick(e) {
    var onClick = props.onClick;

    if (innerLoading) {
      return;
    }

    if (onClick) {
      onClick(e);
    }
  };

  (0, _devWarning["default"])(!(typeof icon === 'string' && icon.length > 2), 'Button', "`icon` is using ReactNode instead of string naming in v4. Please check `".concat(icon, "` at https://ant.design/components/icon"));
  (0, _devWarning["default"])(!(ghost && isUnborderedButtonType(type)), 'Button', "`link` or `text` button can't be a `ghost` button.");
  var prefixCls = getPrefixCls('btn', customizePrefixCls);
  var autoInsertSpace = autoInsertSpaceInButton !== false; // large => lg
  // small => sm

  var sizeCls = '';

  switch (customizeSize || size) {
    case 'large':
      sizeCls = 'lg';
      break;

    case 'small':
      sizeCls = 'sm';
      break;

    default:
      break;
  }

  var iconType = innerLoading ? 'loading' : icon;
  var classes = (0, _classnames["default"])(prefixCls, (_classNames = {}, (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-").concat(type), type), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-").concat(shape), shape), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-").concat(sizeCls), sizeCls), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-icon-only"), !children && children !== 0 && iconType), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-background-ghost"), ghost && !isUnborderedButtonType(type)), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-loading"), innerLoading), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-two-chinese-chars"), hasTwoCNChar && autoInsertSpace), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-block"), block), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-dangerous"), !!danger), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-rtl"), direction === 'rtl'), _classNames), className);
  var iconNode = icon && !innerLoading ? icon : /*#__PURE__*/React.createElement(_LoadingIcon["default"], {
    existIcon: !!icon,
    prefixCls: prefixCls,
    loading: !!innerLoading
  });
  var kids = children || children === 0 ? spaceChildren(children, isNeedInserted() && autoInsertSpace) : null;
  var linkButtonRestProps = (0, _omit["default"])(rest, ['navigate']);

  if (linkButtonRestProps.href !== undefined) {
    return /*#__PURE__*/React.createElement("a", (0, _extends2["default"])({}, linkButtonRestProps, {
      className: classes,
      onClick: handleClick,
      ref: buttonRef
    }), iconNode, kids);
  }

  var buttonNode = /*#__PURE__*/React.createElement("button", (0, _extends2["default"])({}, rest, {
    type: htmlType,
    className: classes,
    onClick: handleClick,
    ref: buttonRef
  }), iconNode, kids);

  if (isUnborderedButtonType(type)) {
    return buttonNode;
  }

  return /*#__PURE__*/React.createElement(_wave["default"], null, buttonNode);
};

var Button = /*#__PURE__*/React.forwardRef(InternalButton);
Button.displayName = 'Button';
Button.Group = _buttonGroup["default"];
Button.__ANT_BUTTON = true;
var _default = Button;
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/button/index.js":
/*!***********************************************!*\
  !*** ./node_modules/antd/lib/button/index.js ***!
  \***********************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _button = _interopRequireDefault(__webpack_require__(/*! ./button */ "./node_modules/antd/lib/button/button.js"));

var _default = _button["default"];
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/modal/ActionButton.js":
/*!*****************************************************!*\
  !*** ./node_modules/antd/lib/modal/ActionButton.js ***!
  \*****************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var _slicedToArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/slicedToArray */ "./node_modules/@babel/runtime/helpers/slicedToArray.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _button = _interopRequireDefault(__webpack_require__(/*! ../button */ "./node_modules/antd/lib/button/index.js"));

var _button2 = __webpack_require__(/*! ../button/button */ "./node_modules/antd/lib/button/button.js");

var ActionButton = function ActionButton(props) {
  var clickedRef = React.useRef(false);
  var ref = React.useRef();

  var _React$useState = React.useState(false),
      _React$useState2 = (0, _slicedToArray2["default"])(_React$useState, 2),
      loading = _React$useState2[0],
      setLoading = _React$useState2[1];

  React.useEffect(function () {
    var timeoutId;

    if (props.autoFocus) {
      var $this = ref.current;
      timeoutId = setTimeout(function () {
        return $this.focus();
      });
    }

    return function () {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, []);

  var handlePromiseOnOk = function handlePromiseOnOk(returnValueOfOnOk) {
    var closeModal = props.closeModal;

    if (!returnValueOfOnOk || !returnValueOfOnOk.then) {
      return;
    }

    setLoading(true);
    returnValueOfOnOk.then(function () {
      // It's unnecessary to set loading=false, for the Modal will be unmounted after close.
      // setState({ loading: false });
      closeModal.apply(void 0, arguments);
    }, function (e) {
      // Emit error when catch promise reject
      // eslint-disable-next-line no-console
      console.error(e); // See: https://github.com/ant-design/ant-design/issues/6183

      setLoading(false);
      clickedRef.current = false;
    });
  };

  var onClick = function onClick() {
    var actionFn = props.actionFn,
        closeModal = props.closeModal;

    if (clickedRef.current) {
      return;
    }

    clickedRef.current = true;

    if (!actionFn) {
      closeModal();
      return;
    }

    var returnValueOfOnOk;

    if (actionFn.length) {
      returnValueOfOnOk = actionFn(closeModal); // https://github.com/ant-design/ant-design/issues/23358

      clickedRef.current = false;
    } else {
      returnValueOfOnOk = actionFn();

      if (!returnValueOfOnOk) {
        closeModal();
        return;
      }
    }

    handlePromiseOnOk(returnValueOfOnOk);
  };

  var type = props.type,
      children = props.children,
      prefixCls = props.prefixCls,
      buttonProps = props.buttonProps;
  return /*#__PURE__*/React.createElement(_button["default"], (0, _extends2["default"])({}, (0, _button2.convertLegacyProps)(type), {
    onClick: onClick,
    loading: loading,
    prefixCls: prefixCls
  }, buttonProps, {
    ref: ref
  }), children);
};

var _default = ActionButton;
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/modal/ConfirmDialog.js":
/*!******************************************************!*\
  !*** ./node_modules/antd/lib/modal/ConfirmDialog.js ***!
  \******************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _defineProperty2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/defineProperty */ "./node_modules/@babel/runtime/helpers/defineProperty.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _classnames = _interopRequireDefault(__webpack_require__(/*! classnames */ "./node_modules/classnames/index.js"));

var _Modal = _interopRequireDefault(__webpack_require__(/*! ./Modal */ "./node_modules/antd/lib/modal/Modal.js"));

var _ActionButton = _interopRequireDefault(__webpack_require__(/*! ./ActionButton */ "./node_modules/antd/lib/modal/ActionButton.js"));

var _devWarning = _interopRequireDefault(__webpack_require__(/*! ../_util/devWarning */ "./node_modules/antd/lib/_util/devWarning.js"));

var _configProvider = _interopRequireDefault(__webpack_require__(/*! ../config-provider */ "./node_modules/antd/lib/config-provider/index.js"));

var ConfirmDialog = function ConfirmDialog(props) {
  var icon = props.icon,
      onCancel = props.onCancel,
      onOk = props.onOk,
      close = props.close,
      zIndex = props.zIndex,
      afterClose = props.afterClose,
      visible = props.visible,
      keyboard = props.keyboard,
      centered = props.centered,
      getContainer = props.getContainer,
      maskStyle = props.maskStyle,
      okText = props.okText,
      okButtonProps = props.okButtonProps,
      cancelText = props.cancelText,
      cancelButtonProps = props.cancelButtonProps,
      direction = props.direction,
      prefixCls = props.prefixCls,
      rootPrefixCls = props.rootPrefixCls,
      bodyStyle = props.bodyStyle,
      _props$closable = props.closable,
      closable = _props$closable === void 0 ? false : _props$closable,
      closeIcon = props.closeIcon,
      modalRender = props.modalRender,
      focusTriggerAfterClose = props.focusTriggerAfterClose;
  (0, _devWarning["default"])(!(typeof icon === 'string' && icon.length > 2), 'Modal', "`icon` is using ReactNode instead of string naming in v4. Please check `".concat(icon, "` at https://ant.design/components/icon")); // 支持传入{ icon: null }来隐藏`Modal.confirm`默认的Icon

  var okType = props.okType || 'primary';
  var contentPrefixCls = "".concat(prefixCls, "-confirm"); // 默认为 true，保持向下兼容

  var okCancel = 'okCancel' in props ? props.okCancel : true;
  var width = props.width || 416;
  var style = props.style || {};
  var mask = props.mask === undefined ? true : props.mask; // 默认为 false，保持旧版默认行为

  var maskClosable = props.maskClosable === undefined ? false : props.maskClosable;
  var autoFocusButton = props.autoFocusButton === null ? false : props.autoFocusButton || 'ok';
  var transitionName = props.transitionName || 'zoom';
  var maskTransitionName = props.maskTransitionName || 'fade';
  var classString = (0, _classnames["default"])(contentPrefixCls, "".concat(contentPrefixCls, "-").concat(props.type), (0, _defineProperty2["default"])({}, "".concat(contentPrefixCls, "-rtl"), direction === 'rtl'), props.className);
  var cancelButton = okCancel && /*#__PURE__*/React.createElement(_ActionButton["default"], {
    actionFn: onCancel,
    closeModal: close,
    autoFocus: autoFocusButton === 'cancel',
    buttonProps: cancelButtonProps,
    prefixCls: "".concat(rootPrefixCls, "-btn")
  }, cancelText);
  return /*#__PURE__*/React.createElement(_Modal["default"], {
    prefixCls: prefixCls,
    className: classString,
    wrapClassName: (0, _classnames["default"])((0, _defineProperty2["default"])({}, "".concat(contentPrefixCls, "-centered"), !!props.centered)),
    onCancel: function onCancel() {
      return close({
        triggerCancel: true
      });
    },
    visible: visible,
    title: "",
    transitionName: transitionName,
    footer: "",
    maskTransitionName: maskTransitionName,
    mask: mask,
    maskClosable: maskClosable,
    maskStyle: maskStyle,
    style: style,
    width: width,
    zIndex: zIndex,
    afterClose: afterClose,
    keyboard: keyboard,
    centered: centered,
    getContainer: getContainer,
    closable: closable,
    closeIcon: closeIcon,
    modalRender: modalRender,
    focusTriggerAfterClose: focusTriggerAfterClose
  }, /*#__PURE__*/React.createElement("div", {
    className: "".concat(contentPrefixCls, "-body-wrapper")
  }, /*#__PURE__*/React.createElement(_configProvider["default"], {
    prefixCls: rootPrefixCls
  }, /*#__PURE__*/React.createElement("div", {
    className: "".concat(contentPrefixCls, "-body"),
    style: bodyStyle
  }, icon, props.title === undefined ? null : /*#__PURE__*/React.createElement("span", {
    className: "".concat(contentPrefixCls, "-title")
  }, props.title), /*#__PURE__*/React.createElement("div", {
    className: "".concat(contentPrefixCls, "-content")
  }, props.content))), /*#__PURE__*/React.createElement("div", {
    className: "".concat(contentPrefixCls, "-btns")
  }, cancelButton, /*#__PURE__*/React.createElement(_ActionButton["default"], {
    type: okType,
    actionFn: onOk,
    closeModal: close,
    autoFocus: autoFocusButton === 'ok',
    buttonProps: okButtonProps,
    prefixCls: "".concat(rootPrefixCls, "-btn")
  }, okText))));
};

var _default = ConfirmDialog;
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/modal/Modal.js":
/*!**********************************************!*\
  !*** ./node_modules/antd/lib/modal/Modal.js ***!
  \**********************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = exports.destroyFns = void 0;

var _defineProperty2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/defineProperty */ "./node_modules/@babel/runtime/helpers/defineProperty.js"));

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _rcDialog = _interopRequireDefault(__webpack_require__(/*! rc-dialog */ "./node_modules/rc-dialog/es/index.js"));

var _classnames = _interopRequireDefault(__webpack_require__(/*! classnames */ "./node_modules/classnames/index.js"));

var _CloseOutlined = _interopRequireDefault(__webpack_require__(/*! @ant-design/icons/CloseOutlined */ "./node_modules/@ant-design/icons/CloseOutlined.js"));

var _useModal = _interopRequireDefault(__webpack_require__(/*! ./useModal */ "./node_modules/antd/lib/modal/useModal/index.js"));

var _locale = __webpack_require__(/*! ./locale */ "./node_modules/antd/lib/modal/locale.js");

var _button = _interopRequireDefault(__webpack_require__(/*! ../button */ "./node_modules/antd/lib/button/index.js"));

var _button2 = __webpack_require__(/*! ../button/button */ "./node_modules/antd/lib/button/button.js");

var _LocaleReceiver = _interopRequireDefault(__webpack_require__(/*! ../locale-provider/LocaleReceiver */ "./node_modules/antd/lib/locale-provider/LocaleReceiver.js"));

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

var mousePosition;
var destroyFns = []; // ref: https://github.com/ant-design/ant-design/issues/15795

exports.destroyFns = destroyFns;

var getClickPosition = function getClickPosition(e) {
  mousePosition = {
    x: e.pageX,
    y: e.pageY
  }; // 100ms 内发生过点击事件，则从点击位置动画展示
  // 否则直接 zoom 展示
  // 这样可以兼容非点击方式展开

  setTimeout(function () {
    mousePosition = null;
  }, 100);
}; // 只有点击事件支持从鼠标位置动画展开


if (typeof window !== 'undefined' && window.document && window.document.documentElement) {
  document.documentElement.addEventListener('click', getClickPosition, true);
}

var Modal = function Modal(props) {
  var _classNames;

  var _React$useContext = React.useContext(_configProvider.ConfigContext),
      getContextPopupContainer = _React$useContext.getPopupContainer,
      getPrefixCls = _React$useContext.getPrefixCls,
      direction = _React$useContext.direction;

  var handleCancel = function handleCancel(e) {
    var onCancel = props.onCancel;

    if (onCancel) {
      onCancel(e);
    }
  };

  var handleOk = function handleOk(e) {
    var onOk = props.onOk;

    if (onOk) {
      onOk(e);
    }
  };

  var renderFooter = function renderFooter(locale) {
    var okText = props.okText,
        okType = props.okType,
        cancelText = props.cancelText,
        confirmLoading = props.confirmLoading;
    return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(_button["default"], (0, _extends2["default"])({
      onClick: handleCancel
    }, props.cancelButtonProps), cancelText || locale.cancelText), /*#__PURE__*/React.createElement(_button["default"], (0, _extends2["default"])({}, (0, _button2.convertLegacyProps)(okType), {
      loading: confirmLoading,
      onClick: handleOk
    }, props.okButtonProps), okText || locale.okText));
  };

  var customizePrefixCls = props.prefixCls,
      footer = props.footer,
      visible = props.visible,
      wrapClassName = props.wrapClassName,
      centered = props.centered,
      getContainer = props.getContainer,
      closeIcon = props.closeIcon,
      _props$focusTriggerAf = props.focusTriggerAfterClose,
      focusTriggerAfterClose = _props$focusTriggerAf === void 0 ? true : _props$focusTriggerAf,
      restProps = __rest(props, ["prefixCls", "footer", "visible", "wrapClassName", "centered", "getContainer", "closeIcon", "focusTriggerAfterClose"]);

  var prefixCls = getPrefixCls('modal', customizePrefixCls);
  var defaultFooter = /*#__PURE__*/React.createElement(_LocaleReceiver["default"], {
    componentName: "Modal",
    defaultLocale: (0, _locale.getConfirmLocale)()
  }, renderFooter);
  var closeIconToRender = /*#__PURE__*/React.createElement("span", {
    className: "".concat(prefixCls, "-close-x")
  }, closeIcon || /*#__PURE__*/React.createElement(_CloseOutlined["default"], {
    className: "".concat(prefixCls, "-close-icon")
  }));
  var wrapClassNameExtended = (0, _classnames["default"])(wrapClassName, (_classNames = {}, (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-centered"), !!centered), (0, _defineProperty2["default"])(_classNames, "".concat(prefixCls, "-wrap-rtl"), direction === 'rtl'), _classNames));
  return /*#__PURE__*/React.createElement(_rcDialog["default"], (0, _extends2["default"])({}, restProps, {
    getContainer: getContainer === undefined ? getContextPopupContainer : getContainer,
    prefixCls: prefixCls,
    wrapClassName: wrapClassNameExtended,
    footer: footer === undefined ? defaultFooter : footer,
    visible: visible,
    mousePosition: mousePosition,
    onClose: handleCancel,
    closeIcon: closeIconToRender,
    focusTriggerAfterClose: focusTriggerAfterClose
  }));
};

Modal.useModal = _useModal["default"];
Modal.defaultProps = {
  width: 520,
  transitionName: 'zoom',
  maskTransitionName: 'fade',
  confirmLoading: false,
  visible: false,
  okType: 'primary'
};
var _default = Modal;
exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/modal/confirm.js":
/*!************************************************!*\
  !*** ./node_modules/antd/lib/modal/confirm.js ***!
  \************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = confirm;
exports.withWarn = withWarn;
exports.withInfo = withInfo;
exports.withSuccess = withSuccess;
exports.withError = withError;
exports.withConfirm = withConfirm;
exports.globalConfig = globalConfig;

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var ReactDOM = _interopRequireWildcard(__webpack_require__(/*! react-dom */ "./node_modules/react-dom/index.js"));

var _InfoCircleOutlined = _interopRequireDefault(__webpack_require__(/*! @ant-design/icons/InfoCircleOutlined */ "./node_modules/@ant-design/icons/InfoCircleOutlined.js"));

var _CheckCircleOutlined = _interopRequireDefault(__webpack_require__(/*! @ant-design/icons/CheckCircleOutlined */ "./node_modules/@ant-design/icons/CheckCircleOutlined.js"));

var _CloseCircleOutlined = _interopRequireDefault(__webpack_require__(/*! @ant-design/icons/CloseCircleOutlined */ "./node_modules/@ant-design/icons/CloseCircleOutlined.js"));

var _ExclamationCircleOutlined = _interopRequireDefault(__webpack_require__(/*! @ant-design/icons/ExclamationCircleOutlined */ "./node_modules/@ant-design/icons/ExclamationCircleOutlined.js"));

var _locale = __webpack_require__(/*! ./locale */ "./node_modules/antd/lib/modal/locale.js");

var _Modal = __webpack_require__(/*! ./Modal */ "./node_modules/antd/lib/modal/Modal.js");

var _ConfirmDialog = _interopRequireDefault(__webpack_require__(/*! ./ConfirmDialog */ "./node_modules/antd/lib/modal/ConfirmDialog.js"));

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

var defaultRootPrefixCls = 'ant';

function getRootPrefixCls() {
  return defaultRootPrefixCls;
}

function confirm(config) {
  var div = document.createElement('div');
  document.body.appendChild(div); // eslint-disable-next-line @typescript-eslint/no-use-before-define

  var currentConfig = (0, _extends2["default"])((0, _extends2["default"])({}, config), {
    close: close,
    visible: true
  });

  function destroy() {
    var unmountResult = ReactDOM.unmountComponentAtNode(div);

    if (unmountResult && div.parentNode) {
      div.parentNode.removeChild(div);
    }

    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    var triggerCancel = args.some(function (param) {
      return param && param.triggerCancel;
    });

    if (config.onCancel && triggerCancel) {
      config.onCancel.apply(config, args);
    }

    for (var i = 0; i < _Modal.destroyFns.length; i++) {
      var fn = _Modal.destroyFns[i]; // eslint-disable-next-line @typescript-eslint/no-use-before-define

      if (fn === close) {
        _Modal.destroyFns.splice(i, 1);

        break;
      }
    }
  }

  function render(_a) {
    var okText = _a.okText,
        cancelText = _a.cancelText,
        prefixCls = _a.prefixCls,
        props = __rest(_a, ["okText", "cancelText", "prefixCls"]);
    /**
     * https://github.com/ant-design/ant-design/issues/23623
     *
     * Sync render blocks React event. Let's make this async.
     */


    setTimeout(function () {
      var runtimeLocale = (0, _locale.getConfirmLocale)();
      ReactDOM.render( /*#__PURE__*/React.createElement(_ConfirmDialog["default"], (0, _extends2["default"])({}, props, {
        prefixCls: prefixCls || "".concat(getRootPrefixCls(), "-modal"),
        rootPrefixCls: getRootPrefixCls(),
        okText: okText || (props.okCancel ? runtimeLocale.okText : runtimeLocale.justOkText),
        cancelText: cancelText || runtimeLocale.cancelText
      })), div);
    });
  }

  function close() {
    var _this = this;

    for (var _len2 = arguments.length, args = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
      args[_key2] = arguments[_key2];
    }

    currentConfig = (0, _extends2["default"])((0, _extends2["default"])({}, currentConfig), {
      visible: false,
      afterClose: function afterClose() {
        if (typeof config.afterClose === 'function') {
          config.afterClose();
        }

        destroy.apply(_this, args);
      }
    });
    render(currentConfig);
  }

  function update(configUpdate) {
    if (typeof configUpdate === 'function') {
      currentConfig = configUpdate(currentConfig);
    } else {
      currentConfig = (0, _extends2["default"])((0, _extends2["default"])({}, currentConfig), configUpdate);
    }

    render(currentConfig);
  }

  render(currentConfig);

  _Modal.destroyFns.push(close);

  return {
    destroy: close,
    update: update
  };
}

function withWarn(props) {
  return (0, _extends2["default"])((0, _extends2["default"])({
    icon: /*#__PURE__*/React.createElement(_ExclamationCircleOutlined["default"], null),
    okCancel: false
  }, props), {
    type: 'warning'
  });
}

function withInfo(props) {
  return (0, _extends2["default"])((0, _extends2["default"])({
    icon: /*#__PURE__*/React.createElement(_InfoCircleOutlined["default"], null),
    okCancel: false
  }, props), {
    type: 'info'
  });
}

function withSuccess(props) {
  return (0, _extends2["default"])((0, _extends2["default"])({
    icon: /*#__PURE__*/React.createElement(_CheckCircleOutlined["default"], null),
    okCancel: false
  }, props), {
    type: 'success'
  });
}

function withError(props) {
  return (0, _extends2["default"])((0, _extends2["default"])({
    icon: /*#__PURE__*/React.createElement(_CloseCircleOutlined["default"], null),
    okCancel: false
  }, props), {
    type: 'error'
  });
}

function withConfirm(props) {
  return (0, _extends2["default"])((0, _extends2["default"])({
    icon: /*#__PURE__*/React.createElement(_ExclamationCircleOutlined["default"], null),
    okCancel: true
  }, props), {
    type: 'confirm'
  });
}

function globalConfig(_ref) {
  var rootPrefixCls = _ref.rootPrefixCls;

  if (rootPrefixCls) {
    defaultRootPrefixCls = rootPrefixCls;
  }
}

/***/ }),

/***/ "./node_modules/antd/lib/modal/useModal/HookModal.js":
/*!***********************************************************!*\
  !*** ./node_modules/antd/lib/modal/useModal/HookModal.js ***!
  \***********************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = void 0;

var _extends2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/extends */ "./node_modules/@babel/runtime/helpers/extends.js"));

var _slicedToArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/slicedToArray */ "./node_modules/@babel/runtime/helpers/slicedToArray.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _ConfirmDialog = _interopRequireDefault(__webpack_require__(/*! ../ConfirmDialog */ "./node_modules/antd/lib/modal/ConfirmDialog.js"));

var _default2 = _interopRequireDefault(__webpack_require__(/*! ../../locale/default */ "./node_modules/antd/lib/locale/default.js"));

var _LocaleReceiver = _interopRequireDefault(__webpack_require__(/*! ../../locale-provider/LocaleReceiver */ "./node_modules/antd/lib/locale-provider/LocaleReceiver.js"));

var _configProvider = __webpack_require__(/*! ../../config-provider */ "./node_modules/antd/lib/config-provider/index.js");

var HookModal = function HookModal(_ref, ref) {
  var afterClose = _ref.afterClose,
      config = _ref.config;

  var _React$useState = React.useState(true),
      _React$useState2 = (0, _slicedToArray2["default"])(_React$useState, 2),
      visible = _React$useState2[0],
      setVisible = _React$useState2[1];

  var _React$useState3 = React.useState(config),
      _React$useState4 = (0, _slicedToArray2["default"])(_React$useState3, 2),
      innerConfig = _React$useState4[0],
      setInnerConfig = _React$useState4[1];

  var _React$useContext = React.useContext(_configProvider.ConfigContext),
      direction = _React$useContext.direction,
      getPrefixCls = _React$useContext.getPrefixCls;

  var prefixCls = getPrefixCls('modal');
  var rootPrefixCls = getPrefixCls();

  function close() {
    setVisible(false);

    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }

    var triggerCancel = args.some(function (param) {
      return param && param.triggerCancel;
    });

    if (innerConfig.onCancel && triggerCancel) {
      innerConfig.onCancel();
    }
  }

  React.useImperativeHandle(ref, function () {
    return {
      destroy: close,
      update: function update(newConfig) {
        setInnerConfig(function (originConfig) {
          return (0, _extends2["default"])((0, _extends2["default"])({}, originConfig), newConfig);
        });
      }
    };
  });
  return /*#__PURE__*/React.createElement(_LocaleReceiver["default"], {
    componentName: "Modal",
    defaultLocale: _default2["default"].Modal
  }, function (modalLocale) {
    return /*#__PURE__*/React.createElement(_ConfirmDialog["default"], (0, _extends2["default"])({
      prefixCls: prefixCls,
      rootPrefixCls: rootPrefixCls
    }, innerConfig, {
      close: close,
      visible: visible,
      afterClose: afterClose,
      okText: innerConfig.okText || (innerConfig.okCancel ? modalLocale.okText : modalLocale.justOkText),
      direction: direction,
      cancelText: innerConfig.cancelText || modalLocale.cancelText
    }));
  });
};

var _default = /*#__PURE__*/React.forwardRef(HookModal);

exports["default"] = _default;

/***/ }),

/***/ "./node_modules/antd/lib/modal/useModal/index.js":
/*!*******************************************************!*\
  !*** ./node_modules/antd/lib/modal/useModal/index.js ***!
  \*******************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";


var _interopRequireWildcard = __webpack_require__(/*! @babel/runtime/helpers/interopRequireWildcard */ "./node_modules/@babel/runtime/helpers/interopRequireWildcard.js");

var _interopRequireDefault = __webpack_require__(/*! @babel/runtime/helpers/interopRequireDefault */ "./node_modules/@babel/runtime/helpers/interopRequireDefault.js");

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports["default"] = useModal;

var _slicedToArray2 = _interopRequireDefault(__webpack_require__(/*! @babel/runtime/helpers/slicedToArray */ "./node_modules/@babel/runtime/helpers/slicedToArray.js"));

var React = _interopRequireWildcard(__webpack_require__(/*! react */ "./node_modules/react/index.js"));

var _usePatchElement3 = _interopRequireDefault(__webpack_require__(/*! ../../_util/hooks/usePatchElement */ "./node_modules/antd/lib/_util/hooks/usePatchElement.js"));

var _HookModal = _interopRequireDefault(__webpack_require__(/*! ./HookModal */ "./node_modules/antd/lib/modal/useModal/HookModal.js"));

var _confirm = __webpack_require__(/*! ../confirm */ "./node_modules/antd/lib/modal/confirm.js");

var uuid = 0;
var ElementsHolder = /*#__PURE__*/React.memo( /*#__PURE__*/React.forwardRef(function (_props, ref) {
  var _usePatchElement = (0, _usePatchElement3["default"])(),
      _usePatchElement2 = (0, _slicedToArray2["default"])(_usePatchElement, 2),
      elements = _usePatchElement2[0],
      patchElement = _usePatchElement2[1];

  React.useImperativeHandle(ref, function () {
    return {
      patchElement: patchElement
    };
  }, []);
  return /*#__PURE__*/React.createElement(React.Fragment, null, elements);
}));

function useModal() {
  var holderRef = React.useRef(null);
  var getConfirmFunc = React.useCallback(function (withFunc) {
    return function hookConfirm(config) {
      var _a;

      uuid += 1;
      var modalRef = /*#__PURE__*/React.createRef();
      var closeFunc;
      var modal = /*#__PURE__*/React.createElement(_HookModal["default"], {
        key: "modal-".concat(uuid),
        config: withFunc(config),
        ref: modalRef,
        afterClose: function afterClose() {
          closeFunc();
        }
      });
      closeFunc = (_a = holderRef.current) === null || _a === void 0 ? void 0 : _a.patchElement(modal);
      return {
        destroy: function destroy() {
          if (modalRef.current) {
            modalRef.current.destroy();
          }
        },
        update: function update(newConfig) {
          if (modalRef.current) {
            modalRef.current.update(newConfig);
          }
        }
      };
    };
  }, []);
  var fns = React.useMemo(function () {
    return {
      info: getConfirmFunc(_confirm.withInfo),
      success: getConfirmFunc(_confirm.withSuccess),
      error: getConfirmFunc(_confirm.withError),
      warning: getConfirmFunc(_confirm.withWarn),
      confirm: getConfirmFunc(_confirm.withConfirm)
    };
  }, []); // eslint-disable-next-line react/jsx-key

  return [fns, /*#__PURE__*/React.createElement(ElementsHolder, {
    ref: holderRef
  })];
}

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2ZyZWVTZWFyY2hSZXN1bHRNb2RhbC50c3giLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9fdXRpbC9ob29rcy91c2VQYXRjaEVsZW1lbnQuanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9fdXRpbC9yYWYuanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9fdXRpbC91bnJlYWNoYWJsZUV4Y2VwdGlvbi5qcyIsIndlYnBhY2s6Ly9fTl9FLy4vbm9kZV9tb2R1bGVzL2FudGQvbGliL191dGlsL3dhdmUuanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9idXR0b24vTG9hZGluZ0ljb24uanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9idXR0b24vYnV0dG9uLWdyb3VwLmpzIiwid2VicGFjazovL19OX0UvLi9ub2RlX21vZHVsZXMvYW50ZC9saWIvYnV0dG9uL2J1dHRvbi5qcyIsIndlYnBhY2s6Ly9fTl9FLy4vbm9kZV9tb2R1bGVzL2FudGQvbGliL2J1dHRvbi9pbmRleC5qcyIsIndlYnBhY2s6Ly9fTl9FLy4vbm9kZV9tb2R1bGVzL2FudGQvbGliL21vZGFsL0FjdGlvbkJ1dHRvbi5qcyIsIndlYnBhY2s6Ly9fTl9FLy4vbm9kZV9tb2R1bGVzL2FudGQvbGliL21vZGFsL0NvbmZpcm1EaWFsb2cuanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9tb2RhbC9Nb2RhbC5qcyIsIndlYnBhY2s6Ly9fTl9FLy4vbm9kZV9tb2R1bGVzL2FudGQvbGliL21vZGFsL2NvbmZpcm0uanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9tb2RhbC91c2VNb2RhbC9Ib29rTW9kYWwuanMiLCJ3ZWJwYWNrOi8vX05fRS8uL25vZGVfbW9kdWxlcy9hbnRkL2xpYi9tb2RhbC91c2VNb2RhbC9pbmRleC5qcyJdLCJuYW1lcyI6WyJvcGVuX2FfbmV3X3RhYiIsInF1ZXJ5Iiwid2luZG93Iiwib3BlbiIsIlNlYXJjaE1vZGFsIiwic2V0TW9kYWxTdGF0ZSIsIm1vZGFsU3RhdGUiLCJzZWFyY2hfcnVuX251bWJlciIsInNlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRTZWFyY2hEYXRhc2V0TmFtZSIsInNldFNlYXJjaFJ1bk51bWJlciIsInJvdXRlciIsInVzZVJvdXRlciIsImRhdGFzZXQiLCJkYXRhc2V0X25hbWUiLCJ1c2VTdGF0ZSIsImRhdGFzZXROYW1lIiwic2V0RGF0YXNldE5hbWUiLCJvcGVuUnVuSW5OZXdUYWIiLCJ0b2dnbGVSdW5Jbk5ld1RhYiIsInJ1biIsInJ1bl9udW1iZXIiLCJydW5OdW1iZXIiLCJzZXRSdW5OdW1iZXIiLCJ1c2VFZmZlY3QiLCJvbkNsb3NpbmciLCJzZWFyY2hIYW5kbGVyIiwibmF2aWdhdGlvbkhhbmRsZXIiLCJzZWFyY2hfYnlfcnVuX251bWJlciIsInNlYXJjaF9ieV9kYXRhc2V0X25hbWUiLCJ1c2VTZWFyY2giLCJyZXN1bHRzX2dyb3VwZWQiLCJzZWFyY2hpbmciLCJpc0xvYWRpbmciLCJlcnJvcnMiLCJvbk9rIiwicGFyYW1zIiwiZm9ybSIsImdldEZpZWxkc1ZhbHVlIiwibmV3X3RhYl9xdWVyeV9wYXJhbXMiLCJxcyIsInN0cmluZ2lmeSIsImdldENoYW5nZWRRdWVyeVBhcmFtcyIsImN1cnJlbnRfcm9vdCIsImxvY2F0aW9uIiwiaHJlZiIsInNwbGl0Iiwic3VibWl0IiwiRm9ybSIsInVzZUZvcm0iLCJ0aGVtZSIsImNvbG9ycyIsInNlY29uZGFyeSIsIm1haW4iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBRUE7QUFJQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7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OztBQUVyRiw2QkFBNkIsbUJBQU8sQ0FBQyxvSEFBOEM7O0FBRW5GO0FBQ0E7QUFDQSxDQUFDO0FBQ0Q7O0FBRUEsNkNBQTZDLG1CQUFPLENBQUMsb0dBQXNDOztBQUUzRixvQ0FBb0MsbUJBQU8sQ0FBQyw0Q0FBTzs7QUFFbkQsK0NBQStDLG1CQUFPLENBQUMsaUdBQW1DOztBQUUxRix3Q0FBd0MsbUJBQU8sQ0FBQyx3RUFBYTs7QUFFN0QsZUFBZSxtQkFBTyxDQUFDLDREQUFZOztBQUVuQztBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQSxHQUFHO0FBQ0g7QUFDQSxDQUFDOztBQUVEO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxPQUFPO0FBQ1A7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsU0FBUztBQUNUO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsR0FBRztBQUNIO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQSxHQUFHLE1BQU07O0FBRVQ7QUFDQTtBQUNBLEdBQUc7QUFDSCxDIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmU0NmM4OWY5MjY5NDhiM2U0MWFhLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgdXNlU3RhdGUsIHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHFzIGZyb20gJ3FzJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XHJcblxyXG5pbXBvcnQge1xyXG4gIFN0eWxlZE1vZGFsLFxyXG4gIFJlc3VsdHNXcmFwcGVyLFxyXG59IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cyc7XHJcbmltcG9ydCB7IHVzZVNlYXJjaCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVNlYXJjaCc7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IFN0eWxlZEJ1dHRvbiB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IFNlbGVjdGVkRGF0YSB9IGZyb20gJy4vc2VsZWN0ZWREYXRhJztcclxuaW1wb3J0IE5hdiBmcm9tICcuLi9OYXYnO1xyXG5pbXBvcnQgeyBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5pbXBvcnQgeyByb290X3VybCB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQgTW9kYWwgZnJvbSAnYW50ZC9saWIvbW9kYWwvTW9kYWwnO1xyXG5cclxuaW50ZXJmYWNlIEZyZWVTZWFjcmhNb2RhbFByb3BzIHtcclxuICBzZXRNb2RhbFN0YXRlKHN0YXRlOiBib29sZWFuKTogdm9pZDtcclxuICBtb2RhbFN0YXRlOiBib29sZWFuO1xyXG4gIHNlYXJjaF9ydW5fbnVtYmVyOiB1bmRlZmluZWQgfCBzdHJpbmc7XHJcbiAgc2VhcmNoX2RhdGFzZXRfbmFtZTogc3RyaW5nIHwgdW5kZWZpbmVkO1xyXG4gIHNldFNlYXJjaERhdGFzZXROYW1lKGRhdGFzZXRfbmFtZTogYW55KTogdm9pZDtcclxuICBzZXRTZWFyY2hSdW5OdW1iZXIocnVuX251bWJlcjogc3RyaW5nKTogdm9pZDtcclxufVxyXG5cclxuY29uc3Qgb3Blbl9hX25ld190YWIgPSAocXVlcnk6IHN0cmluZykgPT4ge1xyXG4gIHdpbmRvdy5vcGVuKHF1ZXJ5LCAnX2JsYW5rJyk7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgU2VhcmNoTW9kYWwgPSAoe1xyXG4gIHNldE1vZGFsU3RhdGUsXHJcbiAgbW9kYWxTdGF0ZSxcclxuICBzZWFyY2hfcnVuX251bWJlcixcclxuICBzZWFyY2hfZGF0YXNldF9uYW1lLFxyXG4gIHNldFNlYXJjaERhdGFzZXROYW1lLFxyXG4gIHNldFNlYXJjaFJ1bk51bWJlcixcclxufTogRnJlZVNlYWNyaE1vZGFsUHJvcHMpID0+IHtcclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XHJcblxyXG4gIGNvbnN0IFtkYXRhc2V0TmFtZSwgc2V0RGF0YXNldE5hbWVdID0gdXNlU3RhdGUoZGF0YXNldCk7XHJcbiAgY29uc3QgW29wZW5SdW5Jbk5ld1RhYiwgdG9nZ2xlUnVuSW5OZXdUYWJdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG4gIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcbiAgY29uc3QgW3J1bk51bWJlciwgc2V0UnVuTnVtYmVyXSA9IHVzZVN0YXRlPHN0cmluZz4ocnVuKTtcclxuXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGNvbnN0IHJ1biA9IHF1ZXJ5LnJ1bl9udW1iZXIgPyBxdWVyeS5ydW5fbnVtYmVyIDogJyc7XHJcbiAgICBjb25zdCBkYXRhc2V0ID0gcXVlcnkuZGF0YXNldF9uYW1lID8gcXVlcnkuZGF0YXNldF9uYW1lIDogJyc7XHJcbiAgICBzZXREYXRhc2V0TmFtZShkYXRhc2V0KTtcclxuICAgIHNldFJ1bk51bWJlcihydW4pO1xyXG4gIH0sIFtxdWVyeS5kYXRhc2V0X25hbWUsIHF1ZXJ5LnJ1bl9udW1iZXJdKTtcclxuXHJcbiAgY29uc3Qgb25DbG9zaW5nID0gKCkgPT4ge1xyXG4gICAgc2V0TW9kYWxTdGF0ZShmYWxzZSk7XHJcbiAgfTtcclxuXHJcbiAgY29uc3Qgc2VhcmNoSGFuZGxlciA9IChydW5fbnVtYmVyOiBzdHJpbmcsIGRhdGFzZXRfbmFtZTogc3RyaW5nKSA9PiB7XHJcbiAgICBzZXREYXRhc2V0TmFtZShkYXRhc2V0X25hbWUpO1xyXG4gICAgc2V0UnVuTnVtYmVyKHJ1bl9udW1iZXIpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IG5hdmlnYXRpb25IYW5kbGVyID0gKFxyXG4gICAgc2VhcmNoX2J5X3J1bl9udW1iZXI6IHN0cmluZyxcclxuICAgIHNlYXJjaF9ieV9kYXRhc2V0X25hbWU6IHN0cmluZ1xyXG4gICkgPT4ge1xyXG4gICAgc2V0U2VhcmNoUnVuTnVtYmVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyKTtcclxuICAgIHNldFNlYXJjaERhdGFzZXROYW1lKHNlYXJjaF9ieV9kYXRhc2V0X25hbWUpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBzZWFyY2hpbmcsIGlzTG9hZGluZywgZXJyb3JzIH0gPSB1c2VTZWFyY2goXHJcbiAgICBzZWFyY2hfcnVuX251bWJlcixcclxuICAgIHNlYXJjaF9kYXRhc2V0X25hbWVcclxuICApO1xyXG5cclxuICBjb25zdCBvbk9rID0gYXN5bmMgKCkgPT4ge1xyXG4gICAgaWYgKG9wZW5SdW5Jbk5ld1RhYikge1xyXG4gICAgICBjb25zdCBwYXJhbXMgPSBmb3JtLmdldEZpZWxkc1ZhbHVlKCk7XHJcbiAgICAgIGNvbnN0IG5ld190YWJfcXVlcnlfcGFyYW1zID0gcXMuc3RyaW5naWZ5KFxyXG4gICAgICAgIGdldENoYW5nZWRRdWVyeVBhcmFtcyhwYXJhbXMsIHF1ZXJ5KVxyXG4gICAgICApO1xyXG4gICAgICAvL3Jvb3QgdXJsIGlzIGVuZHMgd2l0aCBmaXJzdCAnPycuIEkgY2FuJ3QgdXNlIGp1c3Qgcm9vdCB1cmwgZnJvbSBjb25maWcuY29uZmlnLCBiZWNhdXNlXHJcbiAgICAgIC8vaW4gZGV2IGVudiBpdCB1c2UgbG9jYWxob3N0OjgwODEvZHFtL2RldiAodGhpcyBpcyBvbGQgYmFja2VuZCB1cmwgZnJvbSB3aGVyZSBJJ20gZ2V0dGluZyBkYXRhKSxcclxuICAgICAgLy9idXQgSSBuZWVkIGxvY2FsaG9zdDozMDAwXHJcbiAgICAgIGNvbnN0IGN1cnJlbnRfcm9vdCA9IHdpbmRvdy5sb2NhdGlvbi5ocmVmLnNwbGl0KCcvPycpWzBdO1xyXG4gICAgICBvcGVuX2FfbmV3X3RhYihgJHtjdXJyZW50X3Jvb3R9Lz8ke25ld190YWJfcXVlcnlfcGFyYW1zfWApO1xyXG4gICAgfSBlbHNlIHtcclxuICAgICAgYXdhaXQgZm9ybS5zdWJtaXQoKTtcclxuICAgIH1cclxuICAgIG9uQ2xvc2luZygpO1xyXG4gIH07XHJcblxyXG4gIGNvbnN0IFtmb3JtXSA9IEZvcm0udXNlRm9ybSgpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPE1vZGFsXHJcbiAgICAgIHRpdGxlPVwiU2VhcmNoIGRhdGFcIlxyXG4gICAgICB2aXNpYmxlPXttb2RhbFN0YXRlfVxyXG4gICAgICBvbkNhbmNlbD17KCkgPT4gb25DbG9zaW5nKCl9XHJcbiAgICAgIGZvb3Rlcj17W1xyXG4gICAgICAgIDxTdHlsZWRCdXR0b25cclxuICAgICAgICAgIGNvbG9yPXt0aGVtZS5jb2xvcnMuc2Vjb25kYXJ5Lm1haW59XHJcbiAgICAgICAgICBiYWNrZ3JvdW5kPVwid2hpdGVcIlxyXG4gICAgICAgICAga2V5PVwiQ2xvc2VcIlxyXG4gICAgICAgICAgb25DbGljaz17KCkgPT4gb25DbG9zaW5nKCl9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgQ2xvc2VcclxuICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXHJcbiAgICAgICAgPFN0eWxlZEJ1dHRvbiBrZXk9XCJPS1wiIG9uQ2xpY2s9e29uT2t9PlxyXG4gICAgICAgICAgT0tcclxuICAgICAgICA8L1N0eWxlZEJ1dHRvbj4sXHJcbiAgICAgIF19XHJcbiAgICA+XHJcbiAgICAgIHttb2RhbFN0YXRlICYmIChcclxuICAgICAgICA8PlxyXG4gICAgICAgICAgPE5hdlxyXG4gICAgICAgICAgICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyPXtzZWFyY2hfcnVuX251bWJlcn1cclxuICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPXtzZWFyY2hfZGF0YXNldF9uYW1lfVxyXG4gICAgICAgICAgICBkZWZhdWx0RGF0YXNldE5hbWU9e2RhdGFzZXROYW1lfVxyXG4gICAgICAgICAgICBkZWZhdWx0UnVuTnVtYmVyPXtydW5OdW1iZXJ9XHJcbiAgICAgICAgICAgIGhhbmRsZXI9e25hdmlnYXRpb25IYW5kbGVyfVxyXG4gICAgICAgICAgICB0eXBlPVwidG9wXCJcclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8U2VsZWN0ZWREYXRhXHJcbiAgICAgICAgICAgIGZvcm09e2Zvcm19XHJcbiAgICAgICAgICAgIGRhdGFzZXRfbmFtZT17ZGF0YXNldE5hbWV9XHJcbiAgICAgICAgICAgIHJ1bl9udW1iZXI9e3J1bk51bWJlcn1cclxuICAgICAgICAgICAgdG9nZ2xlUnVuSW5OZXdUYWI9e3RvZ2dsZVJ1bkluTmV3VGFifVxyXG4gICAgICAgICAgICBvcGVuUnVuSW5OZXdUYWI9e29wZW5SdW5Jbk5ld1RhYn1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICB7c2VhcmNoaW5nID8gKFxyXG4gICAgICAgICAgICA8UmVzdWx0c1dyYXBwZXI+XHJcbiAgICAgICAgICAgICAgPFNlYXJjaFJlc3VsdHNcclxuICAgICAgICAgICAgICAgIGhhbmRsZXI9e3NlYXJjaEhhbmRsZXJ9XHJcbiAgICAgICAgICAgICAgICBpc0xvYWRpbmc9e2lzTG9hZGluZ31cclxuICAgICAgICAgICAgICAgIHJlc3VsdHNfZ3JvdXBlZD17cmVzdWx0c19ncm91cGVkfVxyXG4gICAgICAgICAgICAgICAgZXJyb3JzPXtlcnJvcnN9XHJcbiAgICAgICAgICAgICAgLz5cclxuICAgICAgICAgICAgPC9SZXN1bHRzV3JhcHBlcj5cclxuICAgICAgICAgICkgOiAoXHJcbiAgICAgICAgICAgIDxSZXN1bHRzV3JhcHBlciAvPlxyXG4gICAgICAgICAgKX1cclxuICAgICAgICA8Lz5cclxuICAgICAgKX1cclxuICAgIDwvTW9kYWw+XHJcbiAgKTtcclxufTtcclxuIiwiXCJ1c2Ugc3RyaWN0XCI7XG5cbnZhciBfaW50ZXJvcFJlcXVpcmVXaWxkY2FyZCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlV2lsZGNhcmRcIik7XG5cbnZhciBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0ID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVEZWZhdWx0XCIpO1xuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHtcbiAgdmFsdWU6IHRydWVcbn0pO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSB1c2VQYXRjaEVsZW1lbnQ7XG5cbnZhciBfdG9Db25zdW1hYmxlQXJyYXkyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy90b0NvbnN1bWFibGVBcnJheVwiKSk7XG5cbnZhciBfc2xpY2VkVG9BcnJheTIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL3NsaWNlZFRvQXJyYXlcIikpO1xuXG52YXIgUmVhY3QgPSBfaW50ZXJvcFJlcXVpcmVXaWxkY2FyZChyZXF1aXJlKFwicmVhY3RcIikpO1xuXG5mdW5jdGlvbiB1c2VQYXRjaEVsZW1lbnQoKSB7XG4gIHZhciBfUmVhY3QkdXNlU3RhdGUgPSBSZWFjdC51c2VTdGF0ZShbXSksXG4gICAgICBfUmVhY3QkdXNlU3RhdGUyID0gKDAsIF9zbGljZWRUb0FycmF5MltcImRlZmF1bHRcIl0pKF9SZWFjdCR1c2VTdGF0ZSwgMiksXG4gICAgICBlbGVtZW50cyA9IF9SZWFjdCR1c2VTdGF0ZTJbMF0sXG4gICAgICBzZXRFbGVtZW50cyA9IF9SZWFjdCR1c2VTdGF0ZTJbMV07XG5cbiAgdmFyIHBhdGNoRWxlbWVudCA9IFJlYWN0LnVzZUNhbGxiYWNrKGZ1bmN0aW9uIChlbGVtZW50KSB7XG4gICAgLy8gYXBwZW5kIGEgbmV3IGVsZW1lbnQgdG8gZWxlbWVudHMgKGFuZCBjcmVhdGUgYSBuZXcgcmVmKVxuICAgIHNldEVsZW1lbnRzKGZ1bmN0aW9uIChvcmlnaW5FbGVtZW50cykge1xuICAgICAgcmV0dXJuIFtdLmNvbmNhdCgoMCwgX3RvQ29uc3VtYWJsZUFycmF5MltcImRlZmF1bHRcIl0pKG9yaWdpbkVsZW1lbnRzKSwgW2VsZW1lbnRdKTtcbiAgICB9KTsgLy8gcmV0dXJuIGEgZnVuY3Rpb24gdGhhdCByZW1vdmVzIHRoZSBuZXcgZWxlbWVudCBvdXQgb2YgZWxlbWVudHMgKGFuZCBjcmVhdGUgYSBuZXcgcmVmKVxuICAgIC8vIGl0IHdvcmtzIGEgbGl0dGxlIGxpa2UgdXNlRWZmZWN0XG5cbiAgICByZXR1cm4gZnVuY3Rpb24gKCkge1xuICAgICAgc2V0RWxlbWVudHMoZnVuY3Rpb24gKG9yaWdpbkVsZW1lbnRzKSB7XG4gICAgICAgIHJldHVybiBvcmlnaW5FbGVtZW50cy5maWx0ZXIoZnVuY3Rpb24gKGVsZSkge1xuICAgICAgICAgIHJldHVybiBlbGUgIT09IGVsZW1lbnQ7XG4gICAgICAgIH0pO1xuICAgICAgfSk7XG4gICAgfTtcbiAgfSwgW10pO1xuICByZXR1cm4gW2VsZW1lbnRzLCBwYXRjaEVsZW1lbnRdO1xufSIsIlwidXNlIHN0cmljdFwiO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlRGVmYXVsdCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlRGVmYXVsdFwiKTtcblxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7XG4gIHZhbHVlOiB0cnVlXG59KTtcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gd3JhcHBlclJhZjtcblxudmFyIF9yYWYgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJyYy11dGlsL2xpYi9yYWZcIikpO1xuXG52YXIgaWQgPSAwO1xudmFyIGlkcyA9IHt9OyAvLyBTdXBwb3J0IGNhbGwgcmFmIHdpdGggZGVsYXkgc3BlY2lmaWVkIGZyYW1lXG5cbmZ1bmN0aW9uIHdyYXBwZXJSYWYoY2FsbGJhY2spIHtcbiAgdmFyIGRlbGF5RnJhbWVzID0gYXJndW1lbnRzLmxlbmd0aCA+IDEgJiYgYXJndW1lbnRzWzFdICE9PSB1bmRlZmluZWQgPyBhcmd1bWVudHNbMV0gOiAxO1xuICB2YXIgbXlJZCA9IGlkKys7XG4gIHZhciByZXN0RnJhbWVzID0gZGVsYXlGcmFtZXM7XG5cbiAgZnVuY3Rpb24gaW50ZXJuYWxDYWxsYmFjaygpIHtcbiAgICByZXN0RnJhbWVzIC09IDE7XG5cbiAgICBpZiAocmVzdEZyYW1lcyA8PSAwKSB7XG4gICAgICBjYWxsYmFjaygpO1xuICAgICAgZGVsZXRlIGlkc1tteUlkXTtcbiAgICB9IGVsc2Uge1xuICAgICAgaWRzW215SWRdID0gKDAsIF9yYWZbXCJkZWZhdWx0XCJdKShpbnRlcm5hbENhbGxiYWNrKTtcbiAgICB9XG4gIH1cblxuICBpZHNbbXlJZF0gPSAoMCwgX3JhZltcImRlZmF1bHRcIl0pKGludGVybmFsQ2FsbGJhY2spO1xuICByZXR1cm4gbXlJZDtcbn1cblxud3JhcHBlclJhZi5jYW5jZWwgPSBmdW5jdGlvbiBjYW5jZWwocGlkKSB7XG4gIGlmIChwaWQgPT09IHVuZGVmaW5lZCkgcmV0dXJuO1xuXG4gIF9yYWZbXCJkZWZhdWx0XCJdLmNhbmNlbChpZHNbcGlkXSk7XG5cbiAgZGVsZXRlIGlkc1twaWRdO1xufTtcblxud3JhcHBlclJhZi5pZHMgPSBpZHM7IC8vIGV4cG9ydCB0aGlzIGZvciB0ZXN0IHVzYWdlIiwiXCJ1c2Ugc3RyaWN0XCI7XG5cbnZhciBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0ID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVEZWZhdWx0XCIpO1xuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHtcbiAgdmFsdWU6IHRydWVcbn0pO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSB2b2lkIDA7XG5cbnZhciBfY2xhc3NDYWxsQ2hlY2syID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9jbGFzc0NhbGxDaGVja1wiKSk7XG5cbnZhciBVbnJlYWNoYWJsZUV4Y2VwdGlvbiA9IGZ1bmN0aW9uIFVucmVhY2hhYmxlRXhjZXB0aW9uKHZhbHVlKSB7XG4gICgwLCBfY2xhc3NDYWxsQ2hlY2syW1wiZGVmYXVsdFwiXSkodGhpcywgVW5yZWFjaGFibGVFeGNlcHRpb24pO1xuICByZXR1cm4gbmV3IEVycm9yKFwidW5yZWFjaGFibGUgY2FzZTogXCIuY29uY2F0KEpTT04uc3RyaW5naWZ5KHZhbHVlKSkpO1xufTtcblxuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSBVbnJlYWNoYWJsZUV4Y2VwdGlvbjsiLCJcInVzZSBzdHJpY3RcIjtcblxudmFyIF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVXaWxkY2FyZFwiKTtcblxudmFyIF9pbnRlcm9wUmVxdWlyZURlZmF1bHQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZURlZmF1bHRcIik7XG5cbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwge1xuICB2YWx1ZTogdHJ1ZVxufSk7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IHZvaWQgMDtcblxudmFyIF9jbGFzc0NhbGxDaGVjazIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2NsYXNzQ2FsbENoZWNrXCIpKTtcblxudmFyIF9jcmVhdGVDbGFzczIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2NyZWF0ZUNsYXNzXCIpKTtcblxudmFyIF9hc3NlcnRUaGlzSW5pdGlhbGl6ZWQyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9hc3NlcnRUaGlzSW5pdGlhbGl6ZWRcIikpO1xuXG52YXIgX2luaGVyaXRzMiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW5oZXJpdHNcIikpO1xuXG52YXIgX2NyZWF0ZVN1cGVyMiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvY3JlYXRlU3VwZXJcIikpO1xuXG52YXIgUmVhY3QgPSBfaW50ZXJvcFJlcXVpcmVXaWxkY2FyZChyZXF1aXJlKFwicmVhY3RcIikpO1xuXG52YXIgX3JlZjIgPSByZXF1aXJlKFwicmMtdXRpbC9saWIvcmVmXCIpO1xuXG52YXIgX3JhZiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4vcmFmXCIpKTtcblxudmFyIF9jb25maWdQcm92aWRlciA9IHJlcXVpcmUoXCIuLi9jb25maWctcHJvdmlkZXJcIik7XG5cbnZhciBfcmVhY3ROb2RlID0gcmVxdWlyZShcIi4vcmVhY3ROb2RlXCIpO1xuXG52YXIgc3R5bGVGb3JQc2V1ZG87IC8vIFdoZXJlIGVsIGlzIHRoZSBET00gZWxlbWVudCB5b3UnZCBsaWtlIHRvIHRlc3QgZm9yIHZpc2liaWxpdHlcblxuZnVuY3Rpb24gaXNIaWRkZW4oZWxlbWVudCkge1xuICBpZiAocHJvY2Vzcy5lbnYuTk9ERV9FTlYgPT09ICd0ZXN0Jykge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIHJldHVybiAhZWxlbWVudCB8fCBlbGVtZW50Lm9mZnNldFBhcmVudCA9PT0gbnVsbCB8fCBlbGVtZW50LmhpZGRlbjtcbn1cblxuZnVuY3Rpb24gaXNOb3RHcmV5KGNvbG9yKSB7XG4gIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBuby11c2VsZXNzLWVzY2FwZVxuICB2YXIgbWF0Y2ggPSAoY29sb3IgfHwgJycpLm1hdGNoKC9yZ2JhP1xcKChcXGQqKSwgKFxcZCopLCAoXFxkKikoLCBbXFxkLl0qKT9cXCkvKTtcblxuICBpZiAobWF0Y2ggJiYgbWF0Y2hbMV0gJiYgbWF0Y2hbMl0gJiYgbWF0Y2hbM10pIHtcbiAgICByZXR1cm4gIShtYXRjaFsxXSA9PT0gbWF0Y2hbMl0gJiYgbWF0Y2hbMl0gPT09IG1hdGNoWzNdKTtcbiAgfVxuXG4gIHJldHVybiB0cnVlO1xufVxuXG52YXIgV2F2ZSA9IC8qI19fUFVSRV9fKi9mdW5jdGlvbiAoX1JlYWN0JENvbXBvbmVudCkge1xuICAoMCwgX2luaGVyaXRzMltcImRlZmF1bHRcIl0pKFdhdmUsIF9SZWFjdCRDb21wb25lbnQpO1xuXG4gIHZhciBfc3VwZXIgPSAoMCwgX2NyZWF0ZVN1cGVyMltcImRlZmF1bHRcIl0pKFdhdmUpO1xuXG4gIGZ1bmN0aW9uIFdhdmUoKSB7XG4gICAgdmFyIF90aGlzO1xuXG4gICAgKDAsIF9jbGFzc0NhbGxDaGVjazJbXCJkZWZhdWx0XCJdKSh0aGlzLCBXYXZlKTtcbiAgICBfdGhpcyA9IF9zdXBlci5hcHBseSh0aGlzLCBhcmd1bWVudHMpO1xuICAgIF90aGlzLmNvbnRhaW5lclJlZiA9IC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVSZWYoKTtcbiAgICBfdGhpcy5hbmltYXRpb25TdGFydCA9IGZhbHNlO1xuICAgIF90aGlzLmRlc3Ryb3llZCA9IGZhbHNlO1xuXG4gICAgX3RoaXMub25DbGljayA9IGZ1bmN0aW9uIChub2RlLCB3YXZlQ29sb3IpIHtcbiAgICAgIGlmICghbm9kZSB8fCBpc0hpZGRlbihub2RlKSB8fCBub2RlLmNsYXNzTmFtZS5pbmRleE9mKCctbGVhdmUnKSA+PSAwKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgdmFyIGluc2VydEV4dHJhTm9kZSA9IF90aGlzLnByb3BzLmluc2VydEV4dHJhTm9kZTtcbiAgICAgIF90aGlzLmV4dHJhTm9kZSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xuXG4gICAgICB2YXIgX2Fzc2VydFRoaXNJbml0aWFsaXplID0gKDAsIF9hc3NlcnRUaGlzSW5pdGlhbGl6ZWQyW1wiZGVmYXVsdFwiXSkoX3RoaXMpLFxuICAgICAgICAgIGV4dHJhTm9kZSA9IF9hc3NlcnRUaGlzSW5pdGlhbGl6ZS5leHRyYU5vZGU7XG5cbiAgICAgIHZhciBnZXRQcmVmaXhDbHMgPSBfdGhpcy5jb250ZXh0LmdldFByZWZpeENscztcbiAgICAgIGV4dHJhTm9kZS5jbGFzc05hbWUgPSBcIlwiLmNvbmNhdChnZXRQcmVmaXhDbHMoJycpLCBcIi1jbGljay1hbmltYXRpbmctbm9kZVwiKTtcblxuICAgICAgdmFyIGF0dHJpYnV0ZU5hbWUgPSBfdGhpcy5nZXRBdHRyaWJ1dGVOYW1lKCk7XG5cbiAgICAgIG5vZGUuc2V0QXR0cmlidXRlKGF0dHJpYnV0ZU5hbWUsICd0cnVlJyk7IC8vIE5vdCB3aGl0ZSBvciB0cmFuc3BhcmVudCBvciBncmV5XG5cbiAgICAgIHN0eWxlRm9yUHNldWRvID0gc3R5bGVGb3JQc2V1ZG8gfHwgZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3R5bGUnKTtcblxuICAgICAgaWYgKHdhdmVDb2xvciAmJiB3YXZlQ29sb3IgIT09ICcjZmZmZmZmJyAmJiB3YXZlQ29sb3IgIT09ICdyZ2IoMjU1LCAyNTUsIDI1NSknICYmIGlzTm90R3JleSh3YXZlQ29sb3IpICYmICEvcmdiYVxcKCg/OlxcZCosICl7M30wXFwpLy50ZXN0KHdhdmVDb2xvcikgJiYgLy8gYW55IHRyYW5zcGFyZW50IHJnYmEgY29sb3JcbiAgICAgIHdhdmVDb2xvciAhPT0gJ3RyYW5zcGFyZW50Jykge1xuICAgICAgICAvLyBBZGQgbm9uY2UgaWYgQ1NQIGV4aXN0XG4gICAgICAgIGlmIChfdGhpcy5jc3AgJiYgX3RoaXMuY3NwLm5vbmNlKSB7XG4gICAgICAgICAgc3R5bGVGb3JQc2V1ZG8ubm9uY2UgPSBfdGhpcy5jc3Aubm9uY2U7XG4gICAgICAgIH1cblxuICAgICAgICBleHRyYU5vZGUuc3R5bGUuYm9yZGVyQ29sb3IgPSB3YXZlQ29sb3I7XG4gICAgICAgIHN0eWxlRm9yUHNldWRvLmlubmVySFRNTCA9IFwiXFxuICAgICAgW1wiLmNvbmNhdChnZXRQcmVmaXhDbHMoJycpLCBcIi1jbGljay1hbmltYXRpbmctd2l0aG91dC1leHRyYS1ub2RlPSd0cnVlJ106OmFmdGVyLCAuXCIpLmNvbmNhdChnZXRQcmVmaXhDbHMoJycpLCBcIi1jbGljay1hbmltYXRpbmctbm9kZSB7XFxuICAgICAgICAtLWFudGQtd2F2ZS1zaGFkb3ctY29sb3I6IFwiKS5jb25jYXQod2F2ZUNvbG9yLCBcIjtcXG4gICAgICB9XCIpO1xuXG4gICAgICAgIGlmICghbm9kZS5vd25lckRvY3VtZW50LmJvZHkuY29udGFpbnMoc3R5bGVGb3JQc2V1ZG8pKSB7XG4gICAgICAgICAgbm9kZS5vd25lckRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQoc3R5bGVGb3JQc2V1ZG8pO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChpbnNlcnRFeHRyYU5vZGUpIHtcbiAgICAgICAgbm9kZS5hcHBlbmRDaGlsZChleHRyYU5vZGUpO1xuICAgICAgfVxuXG4gICAgICBbJ3RyYW5zaXRpb24nLCAnYW5pbWF0aW9uJ10uZm9yRWFjaChmdW5jdGlvbiAobmFtZSkge1xuICAgICAgICBub2RlLmFkZEV2ZW50TGlzdGVuZXIoXCJcIi5jb25jYXQobmFtZSwgXCJzdGFydFwiKSwgX3RoaXMub25UcmFuc2l0aW9uU3RhcnQpO1xuICAgICAgICBub2RlLmFkZEV2ZW50TGlzdGVuZXIoXCJcIi5jb25jYXQobmFtZSwgXCJlbmRcIiksIF90aGlzLm9uVHJhbnNpdGlvbkVuZCk7XG4gICAgICB9KTtcbiAgICB9O1xuXG4gICAgX3RoaXMub25UcmFuc2l0aW9uU3RhcnQgPSBmdW5jdGlvbiAoZSkge1xuICAgICAgaWYgKF90aGlzLmRlc3Ryb3llZCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIHZhciBub2RlID0gX3RoaXMuY29udGFpbmVyUmVmLmN1cnJlbnQ7XG5cbiAgICAgIGlmICghZSB8fCBlLnRhcmdldCAhPT0gbm9kZSB8fCBfdGhpcy5hbmltYXRpb25TdGFydCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIF90aGlzLnJlc2V0RWZmZWN0KG5vZGUpO1xuICAgIH07XG5cbiAgICBfdGhpcy5vblRyYW5zaXRpb25FbmQgPSBmdW5jdGlvbiAoZSkge1xuICAgICAgaWYgKCFlIHx8IGUuYW5pbWF0aW9uTmFtZSAhPT0gJ2ZhZGVFZmZlY3QnKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgX3RoaXMucmVzZXRFZmZlY3QoZS50YXJnZXQpO1xuICAgIH07XG5cbiAgICBfdGhpcy5iaW5kQW5pbWF0aW9uRXZlbnQgPSBmdW5jdGlvbiAobm9kZSkge1xuICAgICAgaWYgKCFub2RlIHx8ICFub2RlLmdldEF0dHJpYnV0ZSB8fCBub2RlLmdldEF0dHJpYnV0ZSgnZGlzYWJsZWQnKSB8fCBub2RlLmNsYXNzTmFtZS5pbmRleE9mKCdkaXNhYmxlZCcpID49IDApIHtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuXG4gICAgICB2YXIgb25DbGljayA9IGZ1bmN0aW9uIG9uQ2xpY2soZSkge1xuICAgICAgICAvLyBGaXggcmFkaW8gYnV0dG9uIGNsaWNrIHR3aWNlXG4gICAgICAgIGlmIChlLnRhcmdldC50YWdOYW1lID09PSAnSU5QVVQnIHx8IGlzSGlkZGVuKGUudGFyZ2V0KSkge1xuICAgICAgICAgIHJldHVybjtcbiAgICAgICAgfVxuXG4gICAgICAgIF90aGlzLnJlc2V0RWZmZWN0KG5vZGUpOyAvLyBHZXQgd2F2ZSBjb2xvciBmcm9tIHRhcmdldFxuXG5cbiAgICAgICAgdmFyIHdhdmVDb2xvciA9IGdldENvbXB1dGVkU3R5bGUobm9kZSkuZ2V0UHJvcGVydHlWYWx1ZSgnYm9yZGVyLXRvcC1jb2xvcicpIHx8IC8vIEZpcmVmb3ggQ29tcGF0aWJsZVxuICAgICAgICBnZXRDb21wdXRlZFN0eWxlKG5vZGUpLmdldFByb3BlcnR5VmFsdWUoJ2JvcmRlci1jb2xvcicpIHx8IGdldENvbXB1dGVkU3R5bGUobm9kZSkuZ2V0UHJvcGVydHlWYWx1ZSgnYmFja2dyb3VuZC1jb2xvcicpO1xuICAgICAgICBfdGhpcy5jbGlja1dhdmVUaW1lb3V0SWQgPSB3aW5kb3cuc2V0VGltZW91dChmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgcmV0dXJuIF90aGlzLm9uQ2xpY2sobm9kZSwgd2F2ZUNvbG9yKTtcbiAgICAgICAgfSwgMCk7XG5cbiAgICAgICAgX3JhZltcImRlZmF1bHRcIl0uY2FuY2VsKF90aGlzLmFuaW1hdGlvblN0YXJ0SWQpO1xuXG4gICAgICAgIF90aGlzLmFuaW1hdGlvblN0YXJ0ID0gdHJ1ZTsgLy8gUmVuZGVyIHRvIHRyaWdnZXIgdHJhbnNpdGlvbiBldmVudCBjb3N0IDMgZnJhbWVzLiBMZXQncyBkZWxheSAxMCBmcmFtZXMgdG8gcmVzZXQgdGhpcy5cblxuICAgICAgICBfdGhpcy5hbmltYXRpb25TdGFydElkID0gKDAsIF9yYWZbXCJkZWZhdWx0XCJdKShmdW5jdGlvbiAoKSB7XG4gICAgICAgICAgX3RoaXMuYW5pbWF0aW9uU3RhcnQgPSBmYWxzZTtcbiAgICAgICAgfSwgMTApO1xuICAgICAgfTtcblxuICAgICAgbm9kZS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsIG9uQ2xpY2ssIHRydWUpO1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgY2FuY2VsOiBmdW5jdGlvbiBjYW5jZWwoKSB7XG4gICAgICAgICAgbm9kZS5yZW1vdmVFdmVudExpc3RlbmVyKCdjbGljaycsIG9uQ2xpY2ssIHRydWUpO1xuICAgICAgICB9XG4gICAgICB9O1xuICAgIH07XG5cbiAgICBfdGhpcy5yZW5kZXJXYXZlID0gZnVuY3Rpb24gKF9yZWYpIHtcbiAgICAgIHZhciBjc3AgPSBfcmVmLmNzcDtcbiAgICAgIHZhciBjaGlsZHJlbiA9IF90aGlzLnByb3BzLmNoaWxkcmVuO1xuICAgICAgX3RoaXMuY3NwID0gY3NwO1xuICAgICAgaWYgKCEgLyojX19QVVJFX18qL1JlYWN0LmlzVmFsaWRFbGVtZW50KGNoaWxkcmVuKSkgcmV0dXJuIGNoaWxkcmVuO1xuICAgICAgdmFyIHJlZiA9IF90aGlzLmNvbnRhaW5lclJlZjtcblxuICAgICAgaWYgKCgwLCBfcmVmMi5zdXBwb3J0UmVmKShjaGlsZHJlbikpIHtcbiAgICAgICAgcmVmID0gKDAsIF9yZWYyLmNvbXBvc2VSZWYpKGNoaWxkcmVuLnJlZiwgX3RoaXMuY29udGFpbmVyUmVmKTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuICgwLCBfcmVhY3ROb2RlLmNsb25lRWxlbWVudCkoY2hpbGRyZW4sIHtcbiAgICAgICAgcmVmOiByZWZcbiAgICAgIH0pO1xuICAgIH07XG5cbiAgICByZXR1cm4gX3RoaXM7XG4gIH1cblxuICAoMCwgX2NyZWF0ZUNsYXNzMltcImRlZmF1bHRcIl0pKFdhdmUsIFt7XG4gICAga2V5OiBcImNvbXBvbmVudERpZE1vdW50XCIsXG4gICAgdmFsdWU6IGZ1bmN0aW9uIGNvbXBvbmVudERpZE1vdW50KCkge1xuICAgICAgdmFyIG5vZGUgPSB0aGlzLmNvbnRhaW5lclJlZi5jdXJyZW50O1xuXG4gICAgICBpZiAoIW5vZGUgfHwgbm9kZS5ub2RlVHlwZSAhPT0gMSkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIHRoaXMuaW5zdGFuY2UgPSB0aGlzLmJpbmRBbmltYXRpb25FdmVudChub2RlKTtcbiAgICB9XG4gIH0sIHtcbiAgICBrZXk6IFwiY29tcG9uZW50V2lsbFVubW91bnRcIixcbiAgICB2YWx1ZTogZnVuY3Rpb24gY29tcG9uZW50V2lsbFVubW91bnQoKSB7XG4gICAgICBpZiAodGhpcy5pbnN0YW5jZSkge1xuICAgICAgICB0aGlzLmluc3RhbmNlLmNhbmNlbCgpO1xuICAgICAgfVxuXG4gICAgICBpZiAodGhpcy5jbGlja1dhdmVUaW1lb3V0SWQpIHtcbiAgICAgICAgY2xlYXJUaW1lb3V0KHRoaXMuY2xpY2tXYXZlVGltZW91dElkKTtcbiAgICAgIH1cblxuICAgICAgdGhpcy5kZXN0cm95ZWQgPSB0cnVlO1xuICAgIH1cbiAgfSwge1xuICAgIGtleTogXCJnZXRBdHRyaWJ1dGVOYW1lXCIsXG4gICAgdmFsdWU6IGZ1bmN0aW9uIGdldEF0dHJpYnV0ZU5hbWUoKSB7XG4gICAgICB2YXIgZ2V0UHJlZml4Q2xzID0gdGhpcy5jb250ZXh0LmdldFByZWZpeENscztcbiAgICAgIHZhciBpbnNlcnRFeHRyYU5vZGUgPSB0aGlzLnByb3BzLmluc2VydEV4dHJhTm9kZTtcbiAgICAgIHJldHVybiBpbnNlcnRFeHRyYU5vZGUgPyBcIlwiLmNvbmNhdChnZXRQcmVmaXhDbHMoJycpLCBcIi1jbGljay1hbmltYXRpbmdcIikgOiBcIlwiLmNvbmNhdChnZXRQcmVmaXhDbHMoJycpLCBcIi1jbGljay1hbmltYXRpbmctd2l0aG91dC1leHRyYS1ub2RlXCIpO1xuICAgIH1cbiAgfSwge1xuICAgIGtleTogXCJyZXNldEVmZmVjdFwiLFxuICAgIHZhbHVlOiBmdW5jdGlvbiByZXNldEVmZmVjdChub2RlKSB7XG4gICAgICB2YXIgX3RoaXMyID0gdGhpcztcblxuICAgICAgaWYgKCFub2RlIHx8IG5vZGUgPT09IHRoaXMuZXh0cmFOb2RlIHx8ICEobm9kZSBpbnN0YW5jZW9mIEVsZW1lbnQpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgdmFyIGluc2VydEV4dHJhTm9kZSA9IHRoaXMucHJvcHMuaW5zZXJ0RXh0cmFOb2RlO1xuICAgICAgdmFyIGF0dHJpYnV0ZU5hbWUgPSB0aGlzLmdldEF0dHJpYnV0ZU5hbWUoKTtcbiAgICAgIG5vZGUuc2V0QXR0cmlidXRlKGF0dHJpYnV0ZU5hbWUsICdmYWxzZScpOyAvLyBlZGdlIGhhcyBidWcgb24gYHJlbW92ZUF0dHJpYnV0ZWAgIzE0NDY2XG5cbiAgICAgIGlmIChzdHlsZUZvclBzZXVkbykge1xuICAgICAgICBzdHlsZUZvclBzZXVkby5pbm5lckhUTUwgPSAnJztcbiAgICAgIH1cblxuICAgICAgaWYgKGluc2VydEV4dHJhTm9kZSAmJiB0aGlzLmV4dHJhTm9kZSAmJiBub2RlLmNvbnRhaW5zKHRoaXMuZXh0cmFOb2RlKSkge1xuICAgICAgICBub2RlLnJlbW92ZUNoaWxkKHRoaXMuZXh0cmFOb2RlKTtcbiAgICAgIH1cblxuICAgICAgWyd0cmFuc2l0aW9uJywgJ2FuaW1hdGlvbiddLmZvckVhY2goZnVuY3Rpb24gKG5hbWUpIHtcbiAgICAgICAgbm9kZS5yZW1vdmVFdmVudExpc3RlbmVyKFwiXCIuY29uY2F0KG5hbWUsIFwic3RhcnRcIiksIF90aGlzMi5vblRyYW5zaXRpb25TdGFydCk7XG4gICAgICAgIG5vZGUucmVtb3ZlRXZlbnRMaXN0ZW5lcihcIlwiLmNvbmNhdChuYW1lLCBcImVuZFwiKSwgX3RoaXMyLm9uVHJhbnNpdGlvbkVuZCk7XG4gICAgICB9KTtcbiAgICB9XG4gIH0sIHtcbiAgICBrZXk6IFwicmVuZGVyXCIsXG4gICAgdmFsdWU6IGZ1bmN0aW9uIHJlbmRlcigpIHtcbiAgICAgIHJldHVybiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfY29uZmlnUHJvdmlkZXIuQ29uZmlnQ29uc3VtZXIsIG51bGwsIHRoaXMucmVuZGVyV2F2ZSk7XG4gICAgfVxuICB9XSk7XG4gIHJldHVybiBXYXZlO1xufShSZWFjdC5Db21wb25lbnQpO1xuXG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IFdhdmU7XG5XYXZlLmNvbnRleHRUeXBlID0gX2NvbmZpZ1Byb3ZpZGVyLkNvbmZpZ0NvbnRleHQ7IiwiXCJ1c2Ugc3RyaWN0XCI7XG5cbnZhciBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0ID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVEZWZhdWx0XCIpO1xuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHtcbiAgdmFsdWU6IHRydWVcbn0pO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSB2b2lkIDA7XG5cbnZhciBfcmVhY3QgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJyZWFjdFwiKSk7XG5cbnZhciBfcmNNb3Rpb24gPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJyYy1tb3Rpb25cIikpO1xuXG52YXIgX0xvYWRpbmdPdXRsaW5lZCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBhbnQtZGVzaWduL2ljb25zL0xvYWRpbmdPdXRsaW5lZFwiKSk7XG5cbnZhciBnZXRDb2xsYXBzZWRXaWR0aCA9IGZ1bmN0aW9uIGdldENvbGxhcHNlZFdpZHRoKCkge1xuICByZXR1cm4ge1xuICAgIHdpZHRoOiAwLFxuICAgIG9wYWNpdHk6IDAsXG4gICAgdHJhbnNmb3JtOiAnc2NhbGUoMCknXG4gIH07XG59O1xuXG52YXIgZ2V0UmVhbFdpZHRoID0gZnVuY3Rpb24gZ2V0UmVhbFdpZHRoKG5vZGUpIHtcbiAgcmV0dXJuIHtcbiAgICB3aWR0aDogbm9kZS5zY3JvbGxXaWR0aCxcbiAgICBvcGFjaXR5OiAxLFxuICAgIHRyYW5zZm9ybTogJ3NjYWxlKDEpJ1xuICB9O1xufTtcblxudmFyIExvYWRpbmdJY29uID0gZnVuY3Rpb24gTG9hZGluZ0ljb24oX3JlZikge1xuICB2YXIgcHJlZml4Q2xzID0gX3JlZi5wcmVmaXhDbHMsXG4gICAgICBsb2FkaW5nID0gX3JlZi5sb2FkaW5nLFxuICAgICAgZXhpc3RJY29uID0gX3JlZi5leGlzdEljb247XG4gIHZhciB2aXNpYmxlID0gISFsb2FkaW5nO1xuXG4gIGlmIChleGlzdEljb24pIHtcbiAgICByZXR1cm4gLyojX19QVVJFX18qL19yZWFjdFtcImRlZmF1bHRcIl0uY3JlYXRlRWxlbWVudChcInNwYW5cIiwge1xuICAgICAgY2xhc3NOYW1lOiBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWxvYWRpbmctaWNvblwiKVxuICAgIH0sIC8qI19fUFVSRV9fKi9fcmVhY3RbXCJkZWZhdWx0XCJdLmNyZWF0ZUVsZW1lbnQoX0xvYWRpbmdPdXRsaW5lZFtcImRlZmF1bHRcIl0sIG51bGwpKTtcbiAgfVxuXG4gIHJldHVybiAvKiNfX1BVUkVfXyovX3JlYWN0W1wiZGVmYXVsdFwiXS5jcmVhdGVFbGVtZW50KF9yY01vdGlvbltcImRlZmF1bHRcIl0sIHtcbiAgICB2aXNpYmxlOiB2aXNpYmxlIC8vIFdlIGRvIG5vdCByZWFsbHkgdXNlIHRoaXMgbW90aW9uTmFtZVxuICAgICxcbiAgICBtb3Rpb25OYW1lOiBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWxvYWRpbmctaWNvbi1tb3Rpb25cIiksXG4gICAgcmVtb3ZlT25MZWF2ZTogdHJ1ZSxcbiAgICBvbkFwcGVhclN0YXJ0OiBnZXRDb2xsYXBzZWRXaWR0aCxcbiAgICBvbkFwcGVhckFjdGl2ZTogZ2V0UmVhbFdpZHRoLFxuICAgIG9uRW50ZXJTdGFydDogZ2V0Q29sbGFwc2VkV2lkdGgsXG4gICAgb25FbnRlckFjdGl2ZTogZ2V0UmVhbFdpZHRoLFxuICAgIG9uTGVhdmVTdGFydDogZ2V0UmVhbFdpZHRoLFxuICAgIG9uTGVhdmVBY3RpdmU6IGdldENvbGxhcHNlZFdpZHRoXG4gIH0sIGZ1bmN0aW9uIChfcmVmMiwgcmVmKSB7XG4gICAgdmFyIGNsYXNzTmFtZSA9IF9yZWYyLmNsYXNzTmFtZSxcbiAgICAgICAgc3R5bGUgPSBfcmVmMi5zdHlsZTtcbiAgICByZXR1cm4gLyojX19QVVJFX18qL19yZWFjdFtcImRlZmF1bHRcIl0uY3JlYXRlRWxlbWVudChcInNwYW5cIiwge1xuICAgICAgY2xhc3NOYW1lOiBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWxvYWRpbmctaWNvblwiKSxcbiAgICAgIHN0eWxlOiBzdHlsZSxcbiAgICAgIHJlZjogcmVmXG4gICAgfSwgLyojX19QVVJFX18qL19yZWFjdFtcImRlZmF1bHRcIl0uY3JlYXRlRWxlbWVudChfTG9hZGluZ091dGxpbmVkW1wiZGVmYXVsdFwiXSwge1xuICAgICAgY2xhc3NOYW1lOiBjbGFzc05hbWVcbiAgICB9KSk7XG4gIH0pO1xufTtcblxudmFyIF9kZWZhdWx0ID0gTG9hZGluZ0ljb247XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IF9kZWZhdWx0OyIsIlwidXNlIHN0cmljdFwiO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkXCIpO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlRGVmYXVsdCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlRGVmYXVsdFwiKTtcblxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7XG4gIHZhbHVlOiB0cnVlXG59KTtcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gdm9pZCAwO1xuXG52YXIgX2V4dGVuZHMyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9leHRlbmRzXCIpKTtcblxudmFyIF9kZWZpbmVQcm9wZXJ0eTIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2RlZmluZVByb3BlcnR5XCIpKTtcblxudmFyIFJlYWN0ID0gX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQocmVxdWlyZShcInJlYWN0XCIpKTtcblxudmFyIF9jbGFzc25hbWVzID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiY2xhc3NuYW1lc1wiKSk7XG5cbnZhciBfY29uZmlnUHJvdmlkZXIgPSByZXF1aXJlKFwiLi4vY29uZmlnLXByb3ZpZGVyXCIpO1xuXG52YXIgX3VucmVhY2hhYmxlRXhjZXB0aW9uID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi4vX3V0aWwvdW5yZWFjaGFibGVFeGNlcHRpb25cIikpO1xuXG52YXIgX19yZXN0ID0gdm9pZCAwICYmICh2b2lkIDApLl9fcmVzdCB8fCBmdW5jdGlvbiAocywgZSkge1xuICB2YXIgdCA9IHt9O1xuXG4gIGZvciAodmFyIHAgaW4gcykge1xuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocywgcCkgJiYgZS5pbmRleE9mKHApIDwgMCkgdFtwXSA9IHNbcF07XG4gIH1cblxuICBpZiAocyAhPSBudWxsICYmIHR5cGVvZiBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzID09PSBcImZ1bmN0aW9uXCIpIGZvciAodmFyIGkgPSAwLCBwID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhzKTsgaSA8IHAubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoZS5pbmRleE9mKHBbaV0pIDwgMCAmJiBPYmplY3QucHJvdG90eXBlLnByb3BlcnR5SXNFbnVtZXJhYmxlLmNhbGwocywgcFtpXSkpIHRbcFtpXV0gPSBzW3BbaV1dO1xuICB9XG4gIHJldHVybiB0O1xufTtcblxudmFyIEJ1dHRvbkdyb3VwID0gZnVuY3Rpb24gQnV0dG9uR3JvdXAocHJvcHMpIHtcbiAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9jb25maWdQcm92aWRlci5Db25maWdDb25zdW1lciwgbnVsbCwgZnVuY3Rpb24gKF9yZWYpIHtcbiAgICB2YXIgX2NsYXNzTmFtZXM7XG5cbiAgICB2YXIgZ2V0UHJlZml4Q2xzID0gX3JlZi5nZXRQcmVmaXhDbHMsXG4gICAgICAgIGRpcmVjdGlvbiA9IF9yZWYuZGlyZWN0aW9uO1xuXG4gICAgdmFyIGN1c3RvbWl6ZVByZWZpeENscyA9IHByb3BzLnByZWZpeENscyxcbiAgICAgICAgc2l6ZSA9IHByb3BzLnNpemUsXG4gICAgICAgIGNsYXNzTmFtZSA9IHByb3BzLmNsYXNzTmFtZSxcbiAgICAgICAgb3RoZXJzID0gX19yZXN0KHByb3BzLCBbXCJwcmVmaXhDbHNcIiwgXCJzaXplXCIsIFwiY2xhc3NOYW1lXCJdKTtcblxuICAgIHZhciBwcmVmaXhDbHMgPSBnZXRQcmVmaXhDbHMoJ2J0bi1ncm91cCcsIGN1c3RvbWl6ZVByZWZpeENscyk7IC8vIGxhcmdlID0+IGxnXG4gICAgLy8gc21hbGwgPT4gc21cblxuICAgIHZhciBzaXplQ2xzID0gJyc7XG5cbiAgICBzd2l0Y2ggKHNpemUpIHtcbiAgICAgIGNhc2UgJ2xhcmdlJzpcbiAgICAgICAgc2l6ZUNscyA9ICdsZyc7XG4gICAgICAgIGJyZWFrO1xuXG4gICAgICBjYXNlICdzbWFsbCc6XG4gICAgICAgIHNpemVDbHMgPSAnc20nO1xuICAgICAgICBicmVhaztcblxuICAgICAgY2FzZSAnbWlkZGxlJzpcbiAgICAgIGNhc2UgdW5kZWZpbmVkOlxuICAgICAgICBicmVhaztcblxuICAgICAgZGVmYXVsdDpcbiAgICAgICAgLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIG5vLWNvbnNvbGVcbiAgICAgICAgY29uc29sZS53YXJuKG5ldyBfdW5yZWFjaGFibGVFeGNlcHRpb25bXCJkZWZhdWx0XCJdKHNpemUpKTtcbiAgICB9XG5cbiAgICB2YXIgY2xhc3NlcyA9ICgwLCBfY2xhc3NuYW1lc1tcImRlZmF1bHRcIl0pKHByZWZpeENscywgKF9jbGFzc05hbWVzID0ge30sICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItXCIpLmNvbmNhdChzaXplQ2xzKSwgc2l6ZUNscyksICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItcnRsXCIpLCBkaXJlY3Rpb24gPT09ICdydGwnKSwgX2NsYXNzTmFtZXMpLCBjbGFzc05hbWUpO1xuICAgIHJldHVybiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChcImRpdlwiLCAoMCwgX2V4dGVuZHMyW1wiZGVmYXVsdFwiXSkoe30sIG90aGVycywge1xuICAgICAgY2xhc3NOYW1lOiBjbGFzc2VzXG4gICAgfSkpO1xuICB9KTtcbn07XG5cbnZhciBfZGVmYXVsdCA9IEJ1dHRvbkdyb3VwO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSBfZGVmYXVsdDsiLCJcInVzZSBzdHJpY3RcIjtcblxudmFyIF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVXaWxkY2FyZFwiKTtcblxudmFyIF9pbnRlcm9wUmVxdWlyZURlZmF1bHQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZURlZmF1bHRcIik7XG5cbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwge1xuICB2YWx1ZTogdHJ1ZVxufSk7XG5leHBvcnRzLmNvbnZlcnRMZWdhY3lQcm9wcyA9IGNvbnZlcnRMZWdhY3lQcm9wcztcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gdm9pZCAwO1xuXG52YXIgX2V4dGVuZHMyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9leHRlbmRzXCIpKTtcblxudmFyIF9kZWZpbmVQcm9wZXJ0eTIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2RlZmluZVByb3BlcnR5XCIpKTtcblxudmFyIF9zbGljZWRUb0FycmF5MiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvc2xpY2VkVG9BcnJheVwiKSk7XG5cbnZhciBfdHlwZW9mMiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvdHlwZW9mXCIpKTtcblxudmFyIFJlYWN0ID0gX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQocmVxdWlyZShcInJlYWN0XCIpKTtcblxudmFyIF9jbGFzc25hbWVzID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiY2xhc3NuYW1lc1wiKSk7XG5cbnZhciBfb21pdCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcInJjLXV0aWwvbGliL29taXRcIikpO1xuXG52YXIgX2J1dHRvbkdyb3VwID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi9idXR0b24tZ3JvdXBcIikpO1xuXG52YXIgX2NvbmZpZ1Byb3ZpZGVyID0gcmVxdWlyZShcIi4uL2NvbmZpZy1wcm92aWRlclwiKTtcblxudmFyIF93YXZlID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi4vX3V0aWwvd2F2ZVwiKSk7XG5cbnZhciBfdHlwZSA9IHJlcXVpcmUoXCIuLi9fdXRpbC90eXBlXCIpO1xuXG52YXIgX2Rldldhcm5pbmcgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCIuLi9fdXRpbC9kZXZXYXJuaW5nXCIpKTtcblxudmFyIF9TaXplQ29udGV4dCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4uL2NvbmZpZy1wcm92aWRlci9TaXplQ29udGV4dFwiKSk7XG5cbnZhciBfTG9hZGluZ0ljb24gPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCIuL0xvYWRpbmdJY29uXCIpKTtcblxudmFyIF9yZWFjdE5vZGUgPSByZXF1aXJlKFwiLi4vX3V0aWwvcmVhY3ROb2RlXCIpO1xuXG52YXIgX19yZXN0ID0gdm9pZCAwICYmICh2b2lkIDApLl9fcmVzdCB8fCBmdW5jdGlvbiAocywgZSkge1xuICB2YXIgdCA9IHt9O1xuXG4gIGZvciAodmFyIHAgaW4gcykge1xuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocywgcCkgJiYgZS5pbmRleE9mKHApIDwgMCkgdFtwXSA9IHNbcF07XG4gIH1cblxuICBpZiAocyAhPSBudWxsICYmIHR5cGVvZiBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzID09PSBcImZ1bmN0aW9uXCIpIGZvciAodmFyIGkgPSAwLCBwID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhzKTsgaSA8IHAubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoZS5pbmRleE9mKHBbaV0pIDwgMCAmJiBPYmplY3QucHJvdG90eXBlLnByb3BlcnR5SXNFbnVtZXJhYmxlLmNhbGwocywgcFtpXSkpIHRbcFtpXV0gPSBzW3BbaV1dO1xuICB9XG4gIHJldHVybiB0O1xufTtcbi8qIGVzbGludC1kaXNhYmxlIHJlYWN0L2J1dHRvbi1oYXMtdHlwZSAqL1xuXG5cbnZhciByeFR3b0NOQ2hhciA9IC9eW1xcdTRlMDAtXFx1OWZhNV17Mn0kLztcbnZhciBpc1R3b0NOQ2hhciA9IHJ4VHdvQ05DaGFyLnRlc3QuYmluZChyeFR3b0NOQ2hhcik7XG5cbmZ1bmN0aW9uIGlzU3RyaW5nKHN0cikge1xuICByZXR1cm4gdHlwZW9mIHN0ciA9PT0gJ3N0cmluZyc7XG59XG5cbmZ1bmN0aW9uIGlzVW5ib3JkZXJlZEJ1dHRvblR5cGUodHlwZSkge1xuICByZXR1cm4gdHlwZSA9PT0gJ3RleHQnIHx8IHR5cGUgPT09ICdsaW5rJztcbn0gLy8gSW5zZXJ0IG9uZSBzcGFjZSBiZXR3ZWVuIHR3byBjaGluZXNlIGNoYXJhY3RlcnMgYXV0b21hdGljYWxseS5cblxuXG5mdW5jdGlvbiBpbnNlcnRTcGFjZShjaGlsZCwgbmVlZEluc2VydGVkKSB7XG4gIC8vIENoZWNrIHRoZSBjaGlsZCBpZiBpcyB1bmRlZmluZWQgb3IgbnVsbC5cbiAgaWYgKGNoaWxkID09IG51bGwpIHtcbiAgICByZXR1cm47XG4gIH1cblxuICB2YXIgU1BBQ0UgPSBuZWVkSW5zZXJ0ZWQgPyAnICcgOiAnJzsgLy8gc3RyaWN0TnVsbENoZWNrcyBvb3BzLlxuXG4gIGlmICh0eXBlb2YgY2hpbGQgIT09ICdzdHJpbmcnICYmIHR5cGVvZiBjaGlsZCAhPT0gJ251bWJlcicgJiYgaXNTdHJpbmcoY2hpbGQudHlwZSkgJiYgaXNUd29DTkNoYXIoY2hpbGQucHJvcHMuY2hpbGRyZW4pKSB7XG4gICAgcmV0dXJuICgwLCBfcmVhY3ROb2RlLmNsb25lRWxlbWVudCkoY2hpbGQsIHtcbiAgICAgIGNoaWxkcmVuOiBjaGlsZC5wcm9wcy5jaGlsZHJlbi5zcGxpdCgnJykuam9pbihTUEFDRSlcbiAgICB9KTtcbiAgfVxuXG4gIGlmICh0eXBlb2YgY2hpbGQgPT09ICdzdHJpbmcnKSB7XG4gICAgaWYgKGlzVHdvQ05DaGFyKGNoaWxkKSkge1xuICAgICAgY2hpbGQgPSBjaGlsZC5zcGxpdCgnJykuam9pbihTUEFDRSk7XG4gICAgfVxuXG4gICAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KFwic3BhblwiLCBudWxsLCBjaGlsZCk7XG4gIH1cblxuICByZXR1cm4gY2hpbGQ7XG59XG5cbmZ1bmN0aW9uIHNwYWNlQ2hpbGRyZW4oY2hpbGRyZW4sIG5lZWRJbnNlcnRlZCkge1xuICB2YXIgaXNQcmV2Q2hpbGRQdXJlID0gZmFsc2U7XG4gIHZhciBjaGlsZExpc3QgPSBbXTtcbiAgUmVhY3QuQ2hpbGRyZW4uZm9yRWFjaChjaGlsZHJlbiwgZnVuY3Rpb24gKGNoaWxkKSB7XG4gICAgdmFyIHR5cGUgPSAoMCwgX3R5cGVvZjJbXCJkZWZhdWx0XCJdKShjaGlsZCk7XG4gICAgdmFyIGlzQ3VycmVudENoaWxkUHVyZSA9IHR5cGUgPT09ICdzdHJpbmcnIHx8IHR5cGUgPT09ICdudW1iZXInO1xuXG4gICAgaWYgKGlzUHJldkNoaWxkUHVyZSAmJiBpc0N1cnJlbnRDaGlsZFB1cmUpIHtcbiAgICAgIHZhciBsYXN0SW5kZXggPSBjaGlsZExpc3QubGVuZ3RoIC0gMTtcbiAgICAgIHZhciBsYXN0Q2hpbGQgPSBjaGlsZExpc3RbbGFzdEluZGV4XTtcbiAgICAgIGNoaWxkTGlzdFtsYXN0SW5kZXhdID0gXCJcIi5jb25jYXQobGFzdENoaWxkKS5jb25jYXQoY2hpbGQpO1xuICAgIH0gZWxzZSB7XG4gICAgICBjaGlsZExpc3QucHVzaChjaGlsZCk7XG4gICAgfVxuXG4gICAgaXNQcmV2Q2hpbGRQdXJlID0gaXNDdXJyZW50Q2hpbGRQdXJlO1xuICB9KTsgLy8gUGFzcyB0byBSZWFjdC5DaGlsZHJlbi5tYXAgdG8gYXV0byBmaWxsIGtleVxuXG4gIHJldHVybiBSZWFjdC5DaGlsZHJlbi5tYXAoY2hpbGRMaXN0LCBmdW5jdGlvbiAoY2hpbGQpIHtcbiAgICByZXR1cm4gaW5zZXJ0U3BhY2UoY2hpbGQsIG5lZWRJbnNlcnRlZCk7XG4gIH0pO1xufVxuXG52YXIgQnV0dG9uVHlwZXMgPSAoMCwgX3R5cGUudHVwbGUpKCdkZWZhdWx0JywgJ3ByaW1hcnknLCAnZ2hvc3QnLCAnZGFzaGVkJywgJ2xpbmsnLCAndGV4dCcpO1xudmFyIEJ1dHRvblNoYXBlcyA9ICgwLCBfdHlwZS50dXBsZSkoJ2NpcmNsZScsICdyb3VuZCcpO1xudmFyIEJ1dHRvbkhUTUxUeXBlcyA9ICgwLCBfdHlwZS50dXBsZSkoJ3N1Ym1pdCcsICdidXR0b24nLCAncmVzZXQnKTtcblxuZnVuY3Rpb24gY29udmVydExlZ2FjeVByb3BzKHR5cGUpIHtcbiAgaWYgKHR5cGUgPT09ICdkYW5nZXInKSB7XG4gICAgcmV0dXJuIHtcbiAgICAgIGRhbmdlcjogdHJ1ZVxuICAgIH07XG4gIH1cblxuICByZXR1cm4ge1xuICAgIHR5cGU6IHR5cGVcbiAgfTtcbn1cblxudmFyIEludGVybmFsQnV0dG9uID0gZnVuY3Rpb24gSW50ZXJuYWxCdXR0b24ocHJvcHMsIHJlZikge1xuICB2YXIgX2NsYXNzTmFtZXM7XG5cbiAgdmFyIF9wcm9wcyRsb2FkaW5nID0gcHJvcHMubG9hZGluZyxcbiAgICAgIGxvYWRpbmcgPSBfcHJvcHMkbG9hZGluZyA9PT0gdm9pZCAwID8gZmFsc2UgOiBfcHJvcHMkbG9hZGluZyxcbiAgICAgIGN1c3RvbWl6ZVByZWZpeENscyA9IHByb3BzLnByZWZpeENscyxcbiAgICAgIHR5cGUgPSBwcm9wcy50eXBlLFxuICAgICAgZGFuZ2VyID0gcHJvcHMuZGFuZ2VyLFxuICAgICAgc2hhcGUgPSBwcm9wcy5zaGFwZSxcbiAgICAgIGN1c3RvbWl6ZVNpemUgPSBwcm9wcy5zaXplLFxuICAgICAgY2xhc3NOYW1lID0gcHJvcHMuY2xhc3NOYW1lLFxuICAgICAgY2hpbGRyZW4gPSBwcm9wcy5jaGlsZHJlbixcbiAgICAgIGljb24gPSBwcm9wcy5pY29uLFxuICAgICAgX3Byb3BzJGdob3N0ID0gcHJvcHMuZ2hvc3QsXG4gICAgICBnaG9zdCA9IF9wcm9wcyRnaG9zdCA9PT0gdm9pZCAwID8gZmFsc2UgOiBfcHJvcHMkZ2hvc3QsXG4gICAgICBfcHJvcHMkYmxvY2sgPSBwcm9wcy5ibG9jayxcbiAgICAgIGJsb2NrID0gX3Byb3BzJGJsb2NrID09PSB2b2lkIDAgPyBmYWxzZSA6IF9wcm9wcyRibG9jayxcbiAgICAgIF9wcm9wcyRodG1sVHlwZSA9IHByb3BzLmh0bWxUeXBlLFxuICAgICAgaHRtbFR5cGUgPSBfcHJvcHMkaHRtbFR5cGUgPT09IHZvaWQgMCA/ICdidXR0b24nIDogX3Byb3BzJGh0bWxUeXBlLFxuICAgICAgcmVzdCA9IF9fcmVzdChwcm9wcywgW1wibG9hZGluZ1wiLCBcInByZWZpeENsc1wiLCBcInR5cGVcIiwgXCJkYW5nZXJcIiwgXCJzaGFwZVwiLCBcInNpemVcIiwgXCJjbGFzc05hbWVcIiwgXCJjaGlsZHJlblwiLCBcImljb25cIiwgXCJnaG9zdFwiLCBcImJsb2NrXCIsIFwiaHRtbFR5cGVcIl0pO1xuXG4gIHZhciBzaXplID0gUmVhY3QudXNlQ29udGV4dChfU2l6ZUNvbnRleHRbXCJkZWZhdWx0XCJdKTtcblxuICB2YXIgX1JlYWN0JHVzZVN0YXRlID0gUmVhY3QudXNlU3RhdGUoISFsb2FkaW5nKSxcbiAgICAgIF9SZWFjdCR1c2VTdGF0ZTIgPSAoMCwgX3NsaWNlZFRvQXJyYXkyW1wiZGVmYXVsdFwiXSkoX1JlYWN0JHVzZVN0YXRlLCAyKSxcbiAgICAgIGlubmVyTG9hZGluZyA9IF9SZWFjdCR1c2VTdGF0ZTJbMF0sXG4gICAgICBzZXRMb2FkaW5nID0gX1JlYWN0JHVzZVN0YXRlMlsxXTtcblxuICB2YXIgX1JlYWN0JHVzZVN0YXRlMyA9IFJlYWN0LnVzZVN0YXRlKGZhbHNlKSxcbiAgICAgIF9SZWFjdCR1c2VTdGF0ZTQgPSAoMCwgX3NsaWNlZFRvQXJyYXkyW1wiZGVmYXVsdFwiXSkoX1JlYWN0JHVzZVN0YXRlMywgMiksXG4gICAgICBoYXNUd29DTkNoYXIgPSBfUmVhY3QkdXNlU3RhdGU0WzBdLFxuICAgICAgc2V0SGFzVHdvQ05DaGFyID0gX1JlYWN0JHVzZVN0YXRlNFsxXTtcblxuICB2YXIgX1JlYWN0JHVzZUNvbnRleHQgPSBSZWFjdC51c2VDb250ZXh0KF9jb25maWdQcm92aWRlci5Db25maWdDb250ZXh0KSxcbiAgICAgIGdldFByZWZpeENscyA9IF9SZWFjdCR1c2VDb250ZXh0LmdldFByZWZpeENscyxcbiAgICAgIGF1dG9JbnNlcnRTcGFjZUluQnV0dG9uID0gX1JlYWN0JHVzZUNvbnRleHQuYXV0b0luc2VydFNwYWNlSW5CdXR0b24sXG4gICAgICBkaXJlY3Rpb24gPSBfUmVhY3QkdXNlQ29udGV4dC5kaXJlY3Rpb247XG5cbiAgdmFyIGJ1dHRvblJlZiA9IHJlZiB8fCAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlUmVmKCk7XG4gIHZhciBkZWxheVRpbWVvdXRSZWYgPSBSZWFjdC51c2VSZWYoKTtcblxuICB2YXIgaXNOZWVkSW5zZXJ0ZWQgPSBmdW5jdGlvbiBpc05lZWRJbnNlcnRlZCgpIHtcbiAgICByZXR1cm4gUmVhY3QuQ2hpbGRyZW4uY291bnQoY2hpbGRyZW4pID09PSAxICYmICFpY29uICYmICFpc1VuYm9yZGVyZWRCdXR0b25UeXBlKHR5cGUpO1xuICB9O1xuXG4gIHZhciBmaXhUd29DTkNoYXIgPSBmdW5jdGlvbiBmaXhUd29DTkNoYXIoKSB7XG4gICAgLy8gRml4IGZvciBIT0MgdXNhZ2UgbGlrZSA8Rm9ybWF0TWVzc2FnZSAvPlxuICAgIGlmICghYnV0dG9uUmVmIHx8ICFidXR0b25SZWYuY3VycmVudCB8fCBhdXRvSW5zZXJ0U3BhY2VJbkJ1dHRvbiA9PT0gZmFsc2UpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB2YXIgYnV0dG9uVGV4dCA9IGJ1dHRvblJlZi5jdXJyZW50LnRleHRDb250ZW50O1xuXG4gICAgaWYgKGlzTmVlZEluc2VydGVkKCkgJiYgaXNUd29DTkNoYXIoYnV0dG9uVGV4dCkpIHtcbiAgICAgIGlmICghaGFzVHdvQ05DaGFyKSB7XG4gICAgICAgIHNldEhhc1R3b0NOQ2hhcih0cnVlKTtcbiAgICAgIH1cbiAgICB9IGVsc2UgaWYgKGhhc1R3b0NOQ2hhcikge1xuICAgICAgc2V0SGFzVHdvQ05DaGFyKGZhbHNlKTtcbiAgICB9XG4gIH07IC8vID09PT09PT09PT09PT09PSBVcGRhdGUgTG9hZGluZyA9PT09PT09PT09PT09PT1cblxuXG4gIHZhciBsb2FkaW5nT3JEZWxheTtcblxuICBpZiAoKDAsIF90eXBlb2YyW1wiZGVmYXVsdFwiXSkobG9hZGluZykgPT09ICdvYmplY3QnICYmIGxvYWRpbmcuZGVsYXkpIHtcbiAgICBsb2FkaW5nT3JEZWxheSA9IGxvYWRpbmcuZGVsYXkgfHwgdHJ1ZTtcbiAgfSBlbHNlIHtcbiAgICBsb2FkaW5nT3JEZWxheSA9ICEhbG9hZGluZztcbiAgfVxuXG4gIFJlYWN0LnVzZUVmZmVjdChmdW5jdGlvbiAoKSB7XG4gICAgY2xlYXJUaW1lb3V0KGRlbGF5VGltZW91dFJlZi5jdXJyZW50KTtcblxuICAgIGlmICh0eXBlb2YgbG9hZGluZ09yRGVsYXkgPT09ICdudW1iZXInKSB7XG4gICAgICBkZWxheVRpbWVvdXRSZWYuY3VycmVudCA9IHdpbmRvdy5zZXRUaW1lb3V0KGZ1bmN0aW9uICgpIHtcbiAgICAgICAgc2V0TG9hZGluZyhsb2FkaW5nT3JEZWxheSk7XG4gICAgICB9LCBsb2FkaW5nT3JEZWxheSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHNldExvYWRpbmcobG9hZGluZ09yRGVsYXkpO1xuICAgIH1cbiAgfSwgW2xvYWRpbmdPckRlbGF5XSk7XG4gIFJlYWN0LnVzZUVmZmVjdChmaXhUd29DTkNoYXIsIFtidXR0b25SZWZdKTtcblxuICB2YXIgaGFuZGxlQ2xpY2sgPSBmdW5jdGlvbiBoYW5kbGVDbGljayhlKSB7XG4gICAgdmFyIG9uQ2xpY2sgPSBwcm9wcy5vbkNsaWNrO1xuXG4gICAgaWYgKGlubmVyTG9hZGluZykge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGlmIChvbkNsaWNrKSB7XG4gICAgICBvbkNsaWNrKGUpO1xuICAgIH1cbiAgfTtcblxuICAoMCwgX2Rldldhcm5pbmdbXCJkZWZhdWx0XCJdKSghKHR5cGVvZiBpY29uID09PSAnc3RyaW5nJyAmJiBpY29uLmxlbmd0aCA+IDIpLCAnQnV0dG9uJywgXCJgaWNvbmAgaXMgdXNpbmcgUmVhY3ROb2RlIGluc3RlYWQgb2Ygc3RyaW5nIG5hbWluZyBpbiB2NC4gUGxlYXNlIGNoZWNrIGBcIi5jb25jYXQoaWNvbiwgXCJgIGF0IGh0dHBzOi8vYW50LmRlc2lnbi9jb21wb25lbnRzL2ljb25cIikpO1xuICAoMCwgX2Rldldhcm5pbmdbXCJkZWZhdWx0XCJdKSghKGdob3N0ICYmIGlzVW5ib3JkZXJlZEJ1dHRvblR5cGUodHlwZSkpLCAnQnV0dG9uJywgXCJgbGlua2Agb3IgYHRleHRgIGJ1dHRvbiBjYW4ndCBiZSBhIGBnaG9zdGAgYnV0dG9uLlwiKTtcbiAgdmFyIHByZWZpeENscyA9IGdldFByZWZpeENscygnYnRuJywgY3VzdG9taXplUHJlZml4Q2xzKTtcbiAgdmFyIGF1dG9JbnNlcnRTcGFjZSA9IGF1dG9JbnNlcnRTcGFjZUluQnV0dG9uICE9PSBmYWxzZTsgLy8gbGFyZ2UgPT4gbGdcbiAgLy8gc21hbGwgPT4gc21cblxuICB2YXIgc2l6ZUNscyA9ICcnO1xuXG4gIHN3aXRjaCAoY3VzdG9taXplU2l6ZSB8fCBzaXplKSB7XG4gICAgY2FzZSAnbGFyZ2UnOlxuICAgICAgc2l6ZUNscyA9ICdsZyc7XG4gICAgICBicmVhaztcblxuICAgIGNhc2UgJ3NtYWxsJzpcbiAgICAgIHNpemVDbHMgPSAnc20nO1xuICAgICAgYnJlYWs7XG5cbiAgICBkZWZhdWx0OlxuICAgICAgYnJlYWs7XG4gIH1cblxuICB2YXIgaWNvblR5cGUgPSBpbm5lckxvYWRpbmcgPyAnbG9hZGluZycgOiBpY29uO1xuICB2YXIgY2xhc3NlcyA9ICgwLCBfY2xhc3NuYW1lc1tcImRlZmF1bHRcIl0pKHByZWZpeENscywgKF9jbGFzc05hbWVzID0ge30sICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItXCIpLmNvbmNhdCh0eXBlKSwgdHlwZSksICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItXCIpLmNvbmNhdChzaGFwZSksIHNoYXBlKSwgKDAsIF9kZWZpbmVQcm9wZXJ0eTJbXCJkZWZhdWx0XCJdKShfY2xhc3NOYW1lcywgXCJcIi5jb25jYXQocHJlZml4Q2xzLCBcIi1cIikuY29uY2F0KHNpemVDbHMpLCBzaXplQ2xzKSwgKDAsIF9kZWZpbmVQcm9wZXJ0eTJbXCJkZWZhdWx0XCJdKShfY2xhc3NOYW1lcywgXCJcIi5jb25jYXQocHJlZml4Q2xzLCBcIi1pY29uLW9ubHlcIiksICFjaGlsZHJlbiAmJiBjaGlsZHJlbiAhPT0gMCAmJiBpY29uVHlwZSksICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItYmFja2dyb3VuZC1naG9zdFwiKSwgZ2hvc3QgJiYgIWlzVW5ib3JkZXJlZEJ1dHRvblR5cGUodHlwZSkpLCAoMCwgX2RlZmluZVByb3BlcnR5MltcImRlZmF1bHRcIl0pKF9jbGFzc05hbWVzLCBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWxvYWRpbmdcIiksIGlubmVyTG9hZGluZyksICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItdHdvLWNoaW5lc2UtY2hhcnNcIiksIGhhc1R3b0NOQ2hhciAmJiBhdXRvSW5zZXJ0U3BhY2UpLCAoMCwgX2RlZmluZVByb3BlcnR5MltcImRlZmF1bHRcIl0pKF9jbGFzc05hbWVzLCBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWJsb2NrXCIpLCBibG9jayksICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItZGFuZ2Vyb3VzXCIpLCAhIWRhbmdlciksICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItcnRsXCIpLCBkaXJlY3Rpb24gPT09ICdydGwnKSwgX2NsYXNzTmFtZXMpLCBjbGFzc05hbWUpO1xuICB2YXIgaWNvbk5vZGUgPSBpY29uICYmICFpbm5lckxvYWRpbmcgPyBpY29uIDogLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0xvYWRpbmdJY29uW1wiZGVmYXVsdFwiXSwge1xuICAgIGV4aXN0SWNvbjogISFpY29uLFxuICAgIHByZWZpeENsczogcHJlZml4Q2xzLFxuICAgIGxvYWRpbmc6ICEhaW5uZXJMb2FkaW5nXG4gIH0pO1xuICB2YXIga2lkcyA9IGNoaWxkcmVuIHx8IGNoaWxkcmVuID09PSAwID8gc3BhY2VDaGlsZHJlbihjaGlsZHJlbiwgaXNOZWVkSW5zZXJ0ZWQoKSAmJiBhdXRvSW5zZXJ0U3BhY2UpIDogbnVsbDtcbiAgdmFyIGxpbmtCdXR0b25SZXN0UHJvcHMgPSAoMCwgX29taXRbXCJkZWZhdWx0XCJdKShyZXN0LCBbJ25hdmlnYXRlJ10pO1xuXG4gIGlmIChsaW5rQnV0dG9uUmVzdFByb3BzLmhyZWYgIT09IHVuZGVmaW5lZCkge1xuICAgIHJldHVybiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChcImFcIiwgKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHt9LCBsaW5rQnV0dG9uUmVzdFByb3BzLCB7XG4gICAgICBjbGFzc05hbWU6IGNsYXNzZXMsXG4gICAgICBvbkNsaWNrOiBoYW5kbGVDbGljayxcbiAgICAgIHJlZjogYnV0dG9uUmVmXG4gICAgfSksIGljb25Ob2RlLCBraWRzKTtcbiAgfVxuXG4gIHZhciBidXR0b25Ob2RlID0gLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoXCJidXR0b25cIiwgKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHt9LCByZXN0LCB7XG4gICAgdHlwZTogaHRtbFR5cGUsXG4gICAgY2xhc3NOYW1lOiBjbGFzc2VzLFxuICAgIG9uQ2xpY2s6IGhhbmRsZUNsaWNrLFxuICAgIHJlZjogYnV0dG9uUmVmXG4gIH0pLCBpY29uTm9kZSwga2lkcyk7XG5cbiAgaWYgKGlzVW5ib3JkZXJlZEJ1dHRvblR5cGUodHlwZSkpIHtcbiAgICByZXR1cm4gYnV0dG9uTm9kZTtcbiAgfVxuXG4gIHJldHVybiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfd2F2ZVtcImRlZmF1bHRcIl0sIG51bGwsIGJ1dHRvbk5vZGUpO1xufTtcblxudmFyIEJ1dHRvbiA9IC8qI19fUFVSRV9fKi9SZWFjdC5mb3J3YXJkUmVmKEludGVybmFsQnV0dG9uKTtcbkJ1dHRvbi5kaXNwbGF5TmFtZSA9ICdCdXR0b24nO1xuQnV0dG9uLkdyb3VwID0gX2J1dHRvbkdyb3VwW1wiZGVmYXVsdFwiXTtcbkJ1dHRvbi5fX0FOVF9CVVRUT04gPSB0cnVlO1xudmFyIF9kZWZhdWx0ID0gQnV0dG9uO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSBfZGVmYXVsdDsiLCJcInVzZSBzdHJpY3RcIjtcblxudmFyIF9pbnRlcm9wUmVxdWlyZURlZmF1bHQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZURlZmF1bHRcIik7XG5cbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwge1xuICB2YWx1ZTogdHJ1ZVxufSk7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IHZvaWQgMDtcblxudmFyIF9idXR0b24gPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCIuL2J1dHRvblwiKSk7XG5cbnZhciBfZGVmYXVsdCA9IF9idXR0b25bXCJkZWZhdWx0XCJdO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSBfZGVmYXVsdDsiLCJcInVzZSBzdHJpY3RcIjtcblxudmFyIF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVXaWxkY2FyZFwiKTtcblxudmFyIF9pbnRlcm9wUmVxdWlyZURlZmF1bHQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZURlZmF1bHRcIik7XG5cbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwge1xuICB2YWx1ZTogdHJ1ZVxufSk7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IHZvaWQgMDtcblxudmFyIF9leHRlbmRzMiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvZXh0ZW5kc1wiKSk7XG5cbnZhciBfc2xpY2VkVG9BcnJheTIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL3NsaWNlZFRvQXJyYXlcIikpO1xuXG52YXIgUmVhY3QgPSBfaW50ZXJvcFJlcXVpcmVXaWxkY2FyZChyZXF1aXJlKFwicmVhY3RcIikpO1xuXG52YXIgX2J1dHRvbiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4uL2J1dHRvblwiKSk7XG5cbnZhciBfYnV0dG9uMiA9IHJlcXVpcmUoXCIuLi9idXR0b24vYnV0dG9uXCIpO1xuXG52YXIgQWN0aW9uQnV0dG9uID0gZnVuY3Rpb24gQWN0aW9uQnV0dG9uKHByb3BzKSB7XG4gIHZhciBjbGlja2VkUmVmID0gUmVhY3QudXNlUmVmKGZhbHNlKTtcbiAgdmFyIHJlZiA9IFJlYWN0LnVzZVJlZigpO1xuXG4gIHZhciBfUmVhY3QkdXNlU3RhdGUgPSBSZWFjdC51c2VTdGF0ZShmYWxzZSksXG4gICAgICBfUmVhY3QkdXNlU3RhdGUyID0gKDAsIF9zbGljZWRUb0FycmF5MltcImRlZmF1bHRcIl0pKF9SZWFjdCR1c2VTdGF0ZSwgMiksXG4gICAgICBsb2FkaW5nID0gX1JlYWN0JHVzZVN0YXRlMlswXSxcbiAgICAgIHNldExvYWRpbmcgPSBfUmVhY3QkdXNlU3RhdGUyWzFdO1xuXG4gIFJlYWN0LnVzZUVmZmVjdChmdW5jdGlvbiAoKSB7XG4gICAgdmFyIHRpbWVvdXRJZDtcblxuICAgIGlmIChwcm9wcy5hdXRvRm9jdXMpIHtcbiAgICAgIHZhciAkdGhpcyA9IHJlZi5jdXJyZW50O1xuICAgICAgdGltZW91dElkID0gc2V0VGltZW91dChmdW5jdGlvbiAoKSB7XG4gICAgICAgIHJldHVybiAkdGhpcy5mb2N1cygpO1xuICAgICAgfSk7XG4gICAgfVxuXG4gICAgcmV0dXJuIGZ1bmN0aW9uICgpIHtcbiAgICAgIGlmICh0aW1lb3V0SWQpIHtcbiAgICAgICAgY2xlYXJUaW1lb3V0KHRpbWVvdXRJZCk7XG4gICAgICB9XG4gICAgfTtcbiAgfSwgW10pO1xuXG4gIHZhciBoYW5kbGVQcm9taXNlT25PayA9IGZ1bmN0aW9uIGhhbmRsZVByb21pc2VPbk9rKHJldHVyblZhbHVlT2ZPbk9rKSB7XG4gICAgdmFyIGNsb3NlTW9kYWwgPSBwcm9wcy5jbG9zZU1vZGFsO1xuXG4gICAgaWYgKCFyZXR1cm5WYWx1ZU9mT25PayB8fCAhcmV0dXJuVmFsdWVPZk9uT2sudGhlbikge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIHNldExvYWRpbmcodHJ1ZSk7XG4gICAgcmV0dXJuVmFsdWVPZk9uT2sudGhlbihmdW5jdGlvbiAoKSB7XG4gICAgICAvLyBJdCdzIHVubmVjZXNzYXJ5IHRvIHNldCBsb2FkaW5nPWZhbHNlLCBmb3IgdGhlIE1vZGFsIHdpbGwgYmUgdW5tb3VudGVkIGFmdGVyIGNsb3NlLlxuICAgICAgLy8gc2V0U3RhdGUoeyBsb2FkaW5nOiBmYWxzZSB9KTtcbiAgICAgIGNsb3NlTW9kYWwuYXBwbHkodm9pZCAwLCBhcmd1bWVudHMpO1xuICAgIH0sIGZ1bmN0aW9uIChlKSB7XG4gICAgICAvLyBFbWl0IGVycm9yIHdoZW4gY2F0Y2ggcHJvbWlzZSByZWplY3RcbiAgICAgIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBuby1jb25zb2xlXG4gICAgICBjb25zb2xlLmVycm9yKGUpOyAvLyBTZWU6IGh0dHBzOi8vZ2l0aHViLmNvbS9hbnQtZGVzaWduL2FudC1kZXNpZ24vaXNzdWVzLzYxODNcblxuICAgICAgc2V0TG9hZGluZyhmYWxzZSk7XG4gICAgICBjbGlja2VkUmVmLmN1cnJlbnQgPSBmYWxzZTtcbiAgICB9KTtcbiAgfTtcblxuICB2YXIgb25DbGljayA9IGZ1bmN0aW9uIG9uQ2xpY2soKSB7XG4gICAgdmFyIGFjdGlvbkZuID0gcHJvcHMuYWN0aW9uRm4sXG4gICAgICAgIGNsb3NlTW9kYWwgPSBwcm9wcy5jbG9zZU1vZGFsO1xuXG4gICAgaWYgKGNsaWNrZWRSZWYuY3VycmVudCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNsaWNrZWRSZWYuY3VycmVudCA9IHRydWU7XG5cbiAgICBpZiAoIWFjdGlvbkZuKSB7XG4gICAgICBjbG9zZU1vZGFsKCk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdmFyIHJldHVyblZhbHVlT2ZPbk9rO1xuXG4gICAgaWYgKGFjdGlvbkZuLmxlbmd0aCkge1xuICAgICAgcmV0dXJuVmFsdWVPZk9uT2sgPSBhY3Rpb25GbihjbG9zZU1vZGFsKTsgLy8gaHR0cHM6Ly9naXRodWIuY29tL2FudC1kZXNpZ24vYW50LWRlc2lnbi9pc3N1ZXMvMjMzNThcblxuICAgICAgY2xpY2tlZFJlZi5jdXJyZW50ID0gZmFsc2U7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVyblZhbHVlT2ZPbk9rID0gYWN0aW9uRm4oKTtcblxuICAgICAgaWYgKCFyZXR1cm5WYWx1ZU9mT25Paykge1xuICAgICAgICBjbG9zZU1vZGFsKCk7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBoYW5kbGVQcm9taXNlT25PayhyZXR1cm5WYWx1ZU9mT25Payk7XG4gIH07XG5cbiAgdmFyIHR5cGUgPSBwcm9wcy50eXBlLFxuICAgICAgY2hpbGRyZW4gPSBwcm9wcy5jaGlsZHJlbixcbiAgICAgIHByZWZpeENscyA9IHByb3BzLnByZWZpeENscyxcbiAgICAgIGJ1dHRvblByb3BzID0gcHJvcHMuYnV0dG9uUHJvcHM7XG4gIHJldHVybiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfYnV0dG9uW1wiZGVmYXVsdFwiXSwgKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHt9LCAoMCwgX2J1dHRvbjIuY29udmVydExlZ2FjeVByb3BzKSh0eXBlKSwge1xuICAgIG9uQ2xpY2s6IG9uQ2xpY2ssXG4gICAgbG9hZGluZzogbG9hZGluZyxcbiAgICBwcmVmaXhDbHM6IHByZWZpeENsc1xuICB9LCBidXR0b25Qcm9wcywge1xuICAgIHJlZjogcmVmXG4gIH0pLCBjaGlsZHJlbik7XG59O1xuXG52YXIgX2RlZmF1bHQgPSBBY3Rpb25CdXR0b247XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IF9kZWZhdWx0OyIsIlwidXNlIHN0cmljdFwiO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkXCIpO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlRGVmYXVsdCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlRGVmYXVsdFwiKTtcblxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7XG4gIHZhbHVlOiB0cnVlXG59KTtcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gdm9pZCAwO1xuXG52YXIgX2RlZmluZVByb3BlcnR5MiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvZGVmaW5lUHJvcGVydHlcIikpO1xuXG52YXIgUmVhY3QgPSBfaW50ZXJvcFJlcXVpcmVXaWxkY2FyZChyZXF1aXJlKFwicmVhY3RcIikpO1xuXG52YXIgX2NsYXNzbmFtZXMgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJjbGFzc25hbWVzXCIpKTtcblxudmFyIF9Nb2RhbCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4vTW9kYWxcIikpO1xuXG52YXIgX0FjdGlvbkJ1dHRvbiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4vQWN0aW9uQnV0dG9uXCIpKTtcblxudmFyIF9kZXZXYXJuaW5nID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi4vX3V0aWwvZGV2V2FybmluZ1wiKSk7XG5cbnZhciBfY29uZmlnUHJvdmlkZXIgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCIuLi9jb25maWctcHJvdmlkZXJcIikpO1xuXG52YXIgQ29uZmlybURpYWxvZyA9IGZ1bmN0aW9uIENvbmZpcm1EaWFsb2cocHJvcHMpIHtcbiAgdmFyIGljb24gPSBwcm9wcy5pY29uLFxuICAgICAgb25DYW5jZWwgPSBwcm9wcy5vbkNhbmNlbCxcbiAgICAgIG9uT2sgPSBwcm9wcy5vbk9rLFxuICAgICAgY2xvc2UgPSBwcm9wcy5jbG9zZSxcbiAgICAgIHpJbmRleCA9IHByb3BzLnpJbmRleCxcbiAgICAgIGFmdGVyQ2xvc2UgPSBwcm9wcy5hZnRlckNsb3NlLFxuICAgICAgdmlzaWJsZSA9IHByb3BzLnZpc2libGUsXG4gICAgICBrZXlib2FyZCA9IHByb3BzLmtleWJvYXJkLFxuICAgICAgY2VudGVyZWQgPSBwcm9wcy5jZW50ZXJlZCxcbiAgICAgIGdldENvbnRhaW5lciA9IHByb3BzLmdldENvbnRhaW5lcixcbiAgICAgIG1hc2tTdHlsZSA9IHByb3BzLm1hc2tTdHlsZSxcbiAgICAgIG9rVGV4dCA9IHByb3BzLm9rVGV4dCxcbiAgICAgIG9rQnV0dG9uUHJvcHMgPSBwcm9wcy5va0J1dHRvblByb3BzLFxuICAgICAgY2FuY2VsVGV4dCA9IHByb3BzLmNhbmNlbFRleHQsXG4gICAgICBjYW5jZWxCdXR0b25Qcm9wcyA9IHByb3BzLmNhbmNlbEJ1dHRvblByb3BzLFxuICAgICAgZGlyZWN0aW9uID0gcHJvcHMuZGlyZWN0aW9uLFxuICAgICAgcHJlZml4Q2xzID0gcHJvcHMucHJlZml4Q2xzLFxuICAgICAgcm9vdFByZWZpeENscyA9IHByb3BzLnJvb3RQcmVmaXhDbHMsXG4gICAgICBib2R5U3R5bGUgPSBwcm9wcy5ib2R5U3R5bGUsXG4gICAgICBfcHJvcHMkY2xvc2FibGUgPSBwcm9wcy5jbG9zYWJsZSxcbiAgICAgIGNsb3NhYmxlID0gX3Byb3BzJGNsb3NhYmxlID09PSB2b2lkIDAgPyBmYWxzZSA6IF9wcm9wcyRjbG9zYWJsZSxcbiAgICAgIGNsb3NlSWNvbiA9IHByb3BzLmNsb3NlSWNvbixcbiAgICAgIG1vZGFsUmVuZGVyID0gcHJvcHMubW9kYWxSZW5kZXIsXG4gICAgICBmb2N1c1RyaWdnZXJBZnRlckNsb3NlID0gcHJvcHMuZm9jdXNUcmlnZ2VyQWZ0ZXJDbG9zZTtcbiAgKDAsIF9kZXZXYXJuaW5nW1wiZGVmYXVsdFwiXSkoISh0eXBlb2YgaWNvbiA9PT0gJ3N0cmluZycgJiYgaWNvbi5sZW5ndGggPiAyKSwgJ01vZGFsJywgXCJgaWNvbmAgaXMgdXNpbmcgUmVhY3ROb2RlIGluc3RlYWQgb2Ygc3RyaW5nIG5hbWluZyBpbiB2NC4gUGxlYXNlIGNoZWNrIGBcIi5jb25jYXQoaWNvbiwgXCJgIGF0IGh0dHBzOi8vYW50LmRlc2lnbi9jb21wb25lbnRzL2ljb25cIikpOyAvLyDmlK/mjIHkvKDlhaV7IGljb246IG51bGwgfeadpemakOiXj2BNb2RhbC5jb25maXJtYOm7mOiupOeahEljb25cblxuICB2YXIgb2tUeXBlID0gcHJvcHMub2tUeXBlIHx8ICdwcmltYXJ5JztcbiAgdmFyIGNvbnRlbnRQcmVmaXhDbHMgPSBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWNvbmZpcm1cIik7IC8vIOm7mOiupOS4uiB0cnVl77yM5L+d5oyB5ZCR5LiL5YW85a65XG5cbiAgdmFyIG9rQ2FuY2VsID0gJ29rQ2FuY2VsJyBpbiBwcm9wcyA/IHByb3BzLm9rQ2FuY2VsIDogdHJ1ZTtcbiAgdmFyIHdpZHRoID0gcHJvcHMud2lkdGggfHwgNDE2O1xuICB2YXIgc3R5bGUgPSBwcm9wcy5zdHlsZSB8fCB7fTtcbiAgdmFyIG1hc2sgPSBwcm9wcy5tYXNrID09PSB1bmRlZmluZWQgPyB0cnVlIDogcHJvcHMubWFzazsgLy8g6buY6K6k5Li6IGZhbHNl77yM5L+d5oyB5pen54mI6buY6K6k6KGM5Li6XG5cbiAgdmFyIG1hc2tDbG9zYWJsZSA9IHByb3BzLm1hc2tDbG9zYWJsZSA9PT0gdW5kZWZpbmVkID8gZmFsc2UgOiBwcm9wcy5tYXNrQ2xvc2FibGU7XG4gIHZhciBhdXRvRm9jdXNCdXR0b24gPSBwcm9wcy5hdXRvRm9jdXNCdXR0b24gPT09IG51bGwgPyBmYWxzZSA6IHByb3BzLmF1dG9Gb2N1c0J1dHRvbiB8fCAnb2snO1xuICB2YXIgdHJhbnNpdGlvbk5hbWUgPSBwcm9wcy50cmFuc2l0aW9uTmFtZSB8fCAnem9vbSc7XG4gIHZhciBtYXNrVHJhbnNpdGlvbk5hbWUgPSBwcm9wcy5tYXNrVHJhbnNpdGlvbk5hbWUgfHwgJ2ZhZGUnO1xuICB2YXIgY2xhc3NTdHJpbmcgPSAoMCwgX2NsYXNzbmFtZXNbXCJkZWZhdWx0XCJdKShjb250ZW50UHJlZml4Q2xzLCBcIlwiLmNvbmNhdChjb250ZW50UHJlZml4Q2xzLCBcIi1cIikuY29uY2F0KHByb3BzLnR5cGUpLCAoMCwgX2RlZmluZVByb3BlcnR5MltcImRlZmF1bHRcIl0pKHt9LCBcIlwiLmNvbmNhdChjb250ZW50UHJlZml4Q2xzLCBcIi1ydGxcIiksIGRpcmVjdGlvbiA9PT0gJ3J0bCcpLCBwcm9wcy5jbGFzc05hbWUpO1xuICB2YXIgY2FuY2VsQnV0dG9uID0gb2tDYW5jZWwgJiYgLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0FjdGlvbkJ1dHRvbltcImRlZmF1bHRcIl0sIHtcbiAgICBhY3Rpb25Gbjogb25DYW5jZWwsXG4gICAgY2xvc2VNb2RhbDogY2xvc2UsXG4gICAgYXV0b0ZvY3VzOiBhdXRvRm9jdXNCdXR0b24gPT09ICdjYW5jZWwnLFxuICAgIGJ1dHRvblByb3BzOiBjYW5jZWxCdXR0b25Qcm9wcyxcbiAgICBwcmVmaXhDbHM6IFwiXCIuY29uY2F0KHJvb3RQcmVmaXhDbHMsIFwiLWJ0blwiKVxuICB9LCBjYW5jZWxUZXh0KTtcbiAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9Nb2RhbFtcImRlZmF1bHRcIl0sIHtcbiAgICBwcmVmaXhDbHM6IHByZWZpeENscyxcbiAgICBjbGFzc05hbWU6IGNsYXNzU3RyaW5nLFxuICAgIHdyYXBDbGFzc05hbWU6ICgwLCBfY2xhc3NuYW1lc1tcImRlZmF1bHRcIl0pKCgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoe30sIFwiXCIuY29uY2F0KGNvbnRlbnRQcmVmaXhDbHMsIFwiLWNlbnRlcmVkXCIpLCAhIXByb3BzLmNlbnRlcmVkKSksXG4gICAgb25DYW5jZWw6IGZ1bmN0aW9uIG9uQ2FuY2VsKCkge1xuICAgICAgcmV0dXJuIGNsb3NlKHtcbiAgICAgICAgdHJpZ2dlckNhbmNlbDogdHJ1ZVxuICAgICAgfSk7XG4gICAgfSxcbiAgICB2aXNpYmxlOiB2aXNpYmxlLFxuICAgIHRpdGxlOiBcIlwiLFxuICAgIHRyYW5zaXRpb25OYW1lOiB0cmFuc2l0aW9uTmFtZSxcbiAgICBmb290ZXI6IFwiXCIsXG4gICAgbWFza1RyYW5zaXRpb25OYW1lOiBtYXNrVHJhbnNpdGlvbk5hbWUsXG4gICAgbWFzazogbWFzayxcbiAgICBtYXNrQ2xvc2FibGU6IG1hc2tDbG9zYWJsZSxcbiAgICBtYXNrU3R5bGU6IG1hc2tTdHlsZSxcbiAgICBzdHlsZTogc3R5bGUsXG4gICAgd2lkdGg6IHdpZHRoLFxuICAgIHpJbmRleDogekluZGV4LFxuICAgIGFmdGVyQ2xvc2U6IGFmdGVyQ2xvc2UsXG4gICAga2V5Ym9hcmQ6IGtleWJvYXJkLFxuICAgIGNlbnRlcmVkOiBjZW50ZXJlZCxcbiAgICBnZXRDb250YWluZXI6IGdldENvbnRhaW5lcixcbiAgICBjbG9zYWJsZTogY2xvc2FibGUsXG4gICAgY2xvc2VJY29uOiBjbG9zZUljb24sXG4gICAgbW9kYWxSZW5kZXI6IG1vZGFsUmVuZGVyLFxuICAgIGZvY3VzVHJpZ2dlckFmdGVyQ2xvc2U6IGZvY3VzVHJpZ2dlckFmdGVyQ2xvc2VcbiAgfSwgLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoXCJkaXZcIiwge1xuICAgIGNsYXNzTmFtZTogXCJcIi5jb25jYXQoY29udGVudFByZWZpeENscywgXCItYm9keS13cmFwcGVyXCIpXG4gIH0sIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9jb25maWdQcm92aWRlcltcImRlZmF1bHRcIl0sIHtcbiAgICBwcmVmaXhDbHM6IHJvb3RQcmVmaXhDbHNcbiAgfSwgLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoXCJkaXZcIiwge1xuICAgIGNsYXNzTmFtZTogXCJcIi5jb25jYXQoY29udGVudFByZWZpeENscywgXCItYm9keVwiKSxcbiAgICBzdHlsZTogYm9keVN0eWxlXG4gIH0sIGljb24sIHByb3BzLnRpdGxlID09PSB1bmRlZmluZWQgPyBudWxsIDogLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoXCJzcGFuXCIsIHtcbiAgICBjbGFzc05hbWU6IFwiXCIuY29uY2F0KGNvbnRlbnRQcmVmaXhDbHMsIFwiLXRpdGxlXCIpXG4gIH0sIHByb3BzLnRpdGxlKSwgLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoXCJkaXZcIiwge1xuICAgIGNsYXNzTmFtZTogXCJcIi5jb25jYXQoY29udGVudFByZWZpeENscywgXCItY29udGVudFwiKVxuICB9LCBwcm9wcy5jb250ZW50KSkpLCAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChcImRpdlwiLCB7XG4gICAgY2xhc3NOYW1lOiBcIlwiLmNvbmNhdChjb250ZW50UHJlZml4Q2xzLCBcIi1idG5zXCIpXG4gIH0sIGNhbmNlbEJ1dHRvbiwgLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0FjdGlvbkJ1dHRvbltcImRlZmF1bHRcIl0sIHtcbiAgICB0eXBlOiBva1R5cGUsXG4gICAgYWN0aW9uRm46IG9uT2ssXG4gICAgY2xvc2VNb2RhbDogY2xvc2UsXG4gICAgYXV0b0ZvY3VzOiBhdXRvRm9jdXNCdXR0b24gPT09ICdvaycsXG4gICAgYnV0dG9uUHJvcHM6IG9rQnV0dG9uUHJvcHMsXG4gICAgcHJlZml4Q2xzOiBcIlwiLmNvbmNhdChyb290UHJlZml4Q2xzLCBcIi1idG5cIilcbiAgfSwgb2tUZXh0KSkpKTtcbn07XG5cbnZhciBfZGVmYXVsdCA9IENvbmZpcm1EaWFsb2c7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IF9kZWZhdWx0OyIsIlwidXNlIHN0cmljdFwiO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkXCIpO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlRGVmYXVsdCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlRGVmYXVsdFwiKTtcblxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7XG4gIHZhbHVlOiB0cnVlXG59KTtcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gZXhwb3J0cy5kZXN0cm95Rm5zID0gdm9pZCAwO1xuXG52YXIgX2RlZmluZVByb3BlcnR5MiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvZGVmaW5lUHJvcGVydHlcIikpO1xuXG52YXIgX2V4dGVuZHMyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9leHRlbmRzXCIpKTtcblxudmFyIFJlYWN0ID0gX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQocmVxdWlyZShcInJlYWN0XCIpKTtcblxudmFyIF9yY0RpYWxvZyA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcInJjLWRpYWxvZ1wiKSk7XG5cbnZhciBfY2xhc3NuYW1lcyA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcImNsYXNzbmFtZXNcIikpO1xuXG52YXIgX0Nsb3NlT3V0bGluZWQgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYW50LWRlc2lnbi9pY29ucy9DbG9zZU91dGxpbmVkXCIpKTtcblxudmFyIF91c2VNb2RhbCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4vdXNlTW9kYWxcIikpO1xuXG52YXIgX2xvY2FsZSA9IHJlcXVpcmUoXCIuL2xvY2FsZVwiKTtcblxudmFyIF9idXR0b24gPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCIuLi9idXR0b25cIikpO1xuXG52YXIgX2J1dHRvbjIgPSByZXF1aXJlKFwiLi4vYnV0dG9uL2J1dHRvblwiKTtcblxudmFyIF9Mb2NhbGVSZWNlaXZlciA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4uL2xvY2FsZS1wcm92aWRlci9Mb2NhbGVSZWNlaXZlclwiKSk7XG5cbnZhciBfY29uZmlnUHJvdmlkZXIgPSByZXF1aXJlKFwiLi4vY29uZmlnLXByb3ZpZGVyXCIpO1xuXG52YXIgX19yZXN0ID0gdm9pZCAwICYmICh2b2lkIDApLl9fcmVzdCB8fCBmdW5jdGlvbiAocywgZSkge1xuICB2YXIgdCA9IHt9O1xuXG4gIGZvciAodmFyIHAgaW4gcykge1xuICAgIGlmIChPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGwocywgcCkgJiYgZS5pbmRleE9mKHApIDwgMCkgdFtwXSA9IHNbcF07XG4gIH1cblxuICBpZiAocyAhPSBudWxsICYmIHR5cGVvZiBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzID09PSBcImZ1bmN0aW9uXCIpIGZvciAodmFyIGkgPSAwLCBwID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhzKTsgaSA8IHAubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAoZS5pbmRleE9mKHBbaV0pIDwgMCAmJiBPYmplY3QucHJvdG90eXBlLnByb3BlcnR5SXNFbnVtZXJhYmxlLmNhbGwocywgcFtpXSkpIHRbcFtpXV0gPSBzW3BbaV1dO1xuICB9XG4gIHJldHVybiB0O1xufTtcblxudmFyIG1vdXNlUG9zaXRpb247XG52YXIgZGVzdHJveUZucyA9IFtdOyAvLyByZWY6IGh0dHBzOi8vZ2l0aHViLmNvbS9hbnQtZGVzaWduL2FudC1kZXNpZ24vaXNzdWVzLzE1Nzk1XG5cbmV4cG9ydHMuZGVzdHJveUZucyA9IGRlc3Ryb3lGbnM7XG5cbnZhciBnZXRDbGlja1Bvc2l0aW9uID0gZnVuY3Rpb24gZ2V0Q2xpY2tQb3NpdGlvbihlKSB7XG4gIG1vdXNlUG9zaXRpb24gPSB7XG4gICAgeDogZS5wYWdlWCxcbiAgICB5OiBlLnBhZ2VZXG4gIH07IC8vIDEwMG1zIOWGheWPkeeUn+i/h+eCueWHu+S6i+S7tu+8jOWImeS7jueCueWHu+S9jee9ruWKqOeUu+WxleekulxuICAvLyDlkKbliJnnm7TmjqUgem9vbSDlsZXnpLpcbiAgLy8g6L+Z5qC35Y+v5Lul5YW85a656Z2e54K55Ye75pa55byP5bGV5byAXG5cbiAgc2V0VGltZW91dChmdW5jdGlvbiAoKSB7XG4gICAgbW91c2VQb3NpdGlvbiA9IG51bGw7XG4gIH0sIDEwMCk7XG59OyAvLyDlj6rmnInngrnlh7vkuovku7bmlK/mjIHku47pvKDmoIfkvY3nva7liqjnlLvlsZXlvIBcblxuXG5pZiAodHlwZW9mIHdpbmRvdyAhPT0gJ3VuZGVmaW5lZCcgJiYgd2luZG93LmRvY3VtZW50ICYmIHdpbmRvdy5kb2N1bWVudC5kb2N1bWVudEVsZW1lbnQpIHtcbiAgZG9jdW1lbnQuZG9jdW1lbnRFbGVtZW50LmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgZ2V0Q2xpY2tQb3NpdGlvbiwgdHJ1ZSk7XG59XG5cbnZhciBNb2RhbCA9IGZ1bmN0aW9uIE1vZGFsKHByb3BzKSB7XG4gIHZhciBfY2xhc3NOYW1lcztcblxuICB2YXIgX1JlYWN0JHVzZUNvbnRleHQgPSBSZWFjdC51c2VDb250ZXh0KF9jb25maWdQcm92aWRlci5Db25maWdDb250ZXh0KSxcbiAgICAgIGdldENvbnRleHRQb3B1cENvbnRhaW5lciA9IF9SZWFjdCR1c2VDb250ZXh0LmdldFBvcHVwQ29udGFpbmVyLFxuICAgICAgZ2V0UHJlZml4Q2xzID0gX1JlYWN0JHVzZUNvbnRleHQuZ2V0UHJlZml4Q2xzLFxuICAgICAgZGlyZWN0aW9uID0gX1JlYWN0JHVzZUNvbnRleHQuZGlyZWN0aW9uO1xuXG4gIHZhciBoYW5kbGVDYW5jZWwgPSBmdW5jdGlvbiBoYW5kbGVDYW5jZWwoZSkge1xuICAgIHZhciBvbkNhbmNlbCA9IHByb3BzLm9uQ2FuY2VsO1xuXG4gICAgaWYgKG9uQ2FuY2VsKSB7XG4gICAgICBvbkNhbmNlbChlKTtcbiAgICB9XG4gIH07XG5cbiAgdmFyIGhhbmRsZU9rID0gZnVuY3Rpb24gaGFuZGxlT2soZSkge1xuICAgIHZhciBvbk9rID0gcHJvcHMub25PaztcblxuICAgIGlmIChvbk9rKSB7XG4gICAgICBvbk9rKGUpO1xuICAgIH1cbiAgfTtcblxuICB2YXIgcmVuZGVyRm9vdGVyID0gZnVuY3Rpb24gcmVuZGVyRm9vdGVyKGxvY2FsZSkge1xuICAgIHZhciBva1RleHQgPSBwcm9wcy5va1RleHQsXG4gICAgICAgIG9rVHlwZSA9IHByb3BzLm9rVHlwZSxcbiAgICAgICAgY2FuY2VsVGV4dCA9IHByb3BzLmNhbmNlbFRleHQsXG4gICAgICAgIGNvbmZpcm1Mb2FkaW5nID0gcHJvcHMuY29uZmlybUxvYWRpbmc7XG4gICAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KFJlYWN0LkZyYWdtZW50LCBudWxsLCAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfYnV0dG9uW1wiZGVmYXVsdFwiXSwgKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHtcbiAgICAgIG9uQ2xpY2s6IGhhbmRsZUNhbmNlbFxuICAgIH0sIHByb3BzLmNhbmNlbEJ1dHRvblByb3BzKSwgY2FuY2VsVGV4dCB8fCBsb2NhbGUuY2FuY2VsVGV4dCksIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9idXR0b25bXCJkZWZhdWx0XCJdLCAoMCwgX2V4dGVuZHMyW1wiZGVmYXVsdFwiXSkoe30sICgwLCBfYnV0dG9uMi5jb252ZXJ0TGVnYWN5UHJvcHMpKG9rVHlwZSksIHtcbiAgICAgIGxvYWRpbmc6IGNvbmZpcm1Mb2FkaW5nLFxuICAgICAgb25DbGljazogaGFuZGxlT2tcbiAgICB9LCBwcm9wcy5va0J1dHRvblByb3BzKSwgb2tUZXh0IHx8IGxvY2FsZS5va1RleHQpKTtcbiAgfTtcblxuICB2YXIgY3VzdG9taXplUHJlZml4Q2xzID0gcHJvcHMucHJlZml4Q2xzLFxuICAgICAgZm9vdGVyID0gcHJvcHMuZm9vdGVyLFxuICAgICAgdmlzaWJsZSA9IHByb3BzLnZpc2libGUsXG4gICAgICB3cmFwQ2xhc3NOYW1lID0gcHJvcHMud3JhcENsYXNzTmFtZSxcbiAgICAgIGNlbnRlcmVkID0gcHJvcHMuY2VudGVyZWQsXG4gICAgICBnZXRDb250YWluZXIgPSBwcm9wcy5nZXRDb250YWluZXIsXG4gICAgICBjbG9zZUljb24gPSBwcm9wcy5jbG9zZUljb24sXG4gICAgICBfcHJvcHMkZm9jdXNUcmlnZ2VyQWYgPSBwcm9wcy5mb2N1c1RyaWdnZXJBZnRlckNsb3NlLFxuICAgICAgZm9jdXNUcmlnZ2VyQWZ0ZXJDbG9zZSA9IF9wcm9wcyRmb2N1c1RyaWdnZXJBZiA9PT0gdm9pZCAwID8gdHJ1ZSA6IF9wcm9wcyRmb2N1c1RyaWdnZXJBZixcbiAgICAgIHJlc3RQcm9wcyA9IF9fcmVzdChwcm9wcywgW1wicHJlZml4Q2xzXCIsIFwiZm9vdGVyXCIsIFwidmlzaWJsZVwiLCBcIndyYXBDbGFzc05hbWVcIiwgXCJjZW50ZXJlZFwiLCBcImdldENvbnRhaW5lclwiLCBcImNsb3NlSWNvblwiLCBcImZvY3VzVHJpZ2dlckFmdGVyQ2xvc2VcIl0pO1xuXG4gIHZhciBwcmVmaXhDbHMgPSBnZXRQcmVmaXhDbHMoJ21vZGFsJywgY3VzdG9taXplUHJlZml4Q2xzKTtcbiAgdmFyIGRlZmF1bHRGb290ZXIgPSAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfTG9jYWxlUmVjZWl2ZXJbXCJkZWZhdWx0XCJdLCB7XG4gICAgY29tcG9uZW50TmFtZTogXCJNb2RhbFwiLFxuICAgIGRlZmF1bHRMb2NhbGU6ICgwLCBfbG9jYWxlLmdldENvbmZpcm1Mb2NhbGUpKClcbiAgfSwgcmVuZGVyRm9vdGVyKTtcbiAgdmFyIGNsb3NlSWNvblRvUmVuZGVyID0gLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoXCJzcGFuXCIsIHtcbiAgICBjbGFzc05hbWU6IFwiXCIuY29uY2F0KHByZWZpeENscywgXCItY2xvc2UteFwiKVxuICB9LCBjbG9zZUljb24gfHwgLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0Nsb3NlT3V0bGluZWRbXCJkZWZhdWx0XCJdLCB7XG4gICAgY2xhc3NOYW1lOiBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLWNsb3NlLWljb25cIilcbiAgfSkpO1xuICB2YXIgd3JhcENsYXNzTmFtZUV4dGVuZGVkID0gKDAsIF9jbGFzc25hbWVzW1wiZGVmYXVsdFwiXSkod3JhcENsYXNzTmFtZSwgKF9jbGFzc05hbWVzID0ge30sICgwLCBfZGVmaW5lUHJvcGVydHkyW1wiZGVmYXVsdFwiXSkoX2NsYXNzTmFtZXMsIFwiXCIuY29uY2F0KHByZWZpeENscywgXCItY2VudGVyZWRcIiksICEhY2VudGVyZWQpLCAoMCwgX2RlZmluZVByb3BlcnR5MltcImRlZmF1bHRcIl0pKF9jbGFzc05hbWVzLCBcIlwiLmNvbmNhdChwcmVmaXhDbHMsIFwiLXdyYXAtcnRsXCIpLCBkaXJlY3Rpb24gPT09ICdydGwnKSwgX2NsYXNzTmFtZXMpKTtcbiAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9yY0RpYWxvZ1tcImRlZmF1bHRcIl0sICgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7fSwgcmVzdFByb3BzLCB7XG4gICAgZ2V0Q29udGFpbmVyOiBnZXRDb250YWluZXIgPT09IHVuZGVmaW5lZCA/IGdldENvbnRleHRQb3B1cENvbnRhaW5lciA6IGdldENvbnRhaW5lcixcbiAgICBwcmVmaXhDbHM6IHByZWZpeENscyxcbiAgICB3cmFwQ2xhc3NOYW1lOiB3cmFwQ2xhc3NOYW1lRXh0ZW5kZWQsXG4gICAgZm9vdGVyOiBmb290ZXIgPT09IHVuZGVmaW5lZCA/IGRlZmF1bHRGb290ZXIgOiBmb290ZXIsXG4gICAgdmlzaWJsZTogdmlzaWJsZSxcbiAgICBtb3VzZVBvc2l0aW9uOiBtb3VzZVBvc2l0aW9uLFxuICAgIG9uQ2xvc2U6IGhhbmRsZUNhbmNlbCxcbiAgICBjbG9zZUljb246IGNsb3NlSWNvblRvUmVuZGVyLFxuICAgIGZvY3VzVHJpZ2dlckFmdGVyQ2xvc2U6IGZvY3VzVHJpZ2dlckFmdGVyQ2xvc2VcbiAgfSkpO1xufTtcblxuTW9kYWwudXNlTW9kYWwgPSBfdXNlTW9kYWxbXCJkZWZhdWx0XCJdO1xuTW9kYWwuZGVmYXVsdFByb3BzID0ge1xuICB3aWR0aDogNTIwLFxuICB0cmFuc2l0aW9uTmFtZTogJ3pvb20nLFxuICBtYXNrVHJhbnNpdGlvbk5hbWU6ICdmYWRlJyxcbiAgY29uZmlybUxvYWRpbmc6IGZhbHNlLFxuICB2aXNpYmxlOiBmYWxzZSxcbiAgb2tUeXBlOiAncHJpbWFyeSdcbn07XG52YXIgX2RlZmF1bHQgPSBNb2RhbDtcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gX2RlZmF1bHQ7IiwiXCJ1c2Ugc3RyaWN0XCI7XG5cbnZhciBfaW50ZXJvcFJlcXVpcmVXaWxkY2FyZCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlV2lsZGNhcmRcIik7XG5cbnZhciBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0ID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVEZWZhdWx0XCIpO1xuXG5PYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgXCJfX2VzTW9kdWxlXCIsIHtcbiAgdmFsdWU6IHRydWVcbn0pO1xuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSBjb25maXJtO1xuZXhwb3J0cy53aXRoV2FybiA9IHdpdGhXYXJuO1xuZXhwb3J0cy53aXRoSW5mbyA9IHdpdGhJbmZvO1xuZXhwb3J0cy53aXRoU3VjY2VzcyA9IHdpdGhTdWNjZXNzO1xuZXhwb3J0cy53aXRoRXJyb3IgPSB3aXRoRXJyb3I7XG5leHBvcnRzLndpdGhDb25maXJtID0gd2l0aENvbmZpcm07XG5leHBvcnRzLmdsb2JhbENvbmZpZyA9IGdsb2JhbENvbmZpZztcblxudmFyIF9leHRlbmRzMiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvZXh0ZW5kc1wiKSk7XG5cbnZhciBSZWFjdCA9IF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkKHJlcXVpcmUoXCJyZWFjdFwiKSk7XG5cbnZhciBSZWFjdERPTSA9IF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkKHJlcXVpcmUoXCJyZWFjdC1kb21cIikpO1xuXG52YXIgX0luZm9DaXJjbGVPdXRsaW5lZCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBhbnQtZGVzaWduL2ljb25zL0luZm9DaXJjbGVPdXRsaW5lZFwiKSk7XG5cbnZhciBfQ2hlY2tDaXJjbGVPdXRsaW5lZCA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBhbnQtZGVzaWduL2ljb25zL0NoZWNrQ2lyY2xlT3V0bGluZWRcIikpO1xuXG52YXIgX0Nsb3NlQ2lyY2xlT3V0bGluZWQgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCJAYW50LWRlc2lnbi9pY29ucy9DbG9zZUNpcmNsZU91dGxpbmVkXCIpKTtcblxudmFyIF9FeGNsYW1hdGlvbkNpcmNsZU91dGxpbmVkID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGFudC1kZXNpZ24vaWNvbnMvRXhjbGFtYXRpb25DaXJjbGVPdXRsaW5lZFwiKSk7XG5cbnZhciBfbG9jYWxlID0gcmVxdWlyZShcIi4vbG9jYWxlXCIpO1xuXG52YXIgX01vZGFsID0gcmVxdWlyZShcIi4vTW9kYWxcIik7XG5cbnZhciBfQ29uZmlybURpYWxvZyA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4vQ29uZmlybURpYWxvZ1wiKSk7XG5cbnZhciBfX3Jlc3QgPSB2b2lkIDAgJiYgKHZvaWQgMCkuX19yZXN0IHx8IGZ1bmN0aW9uIChzLCBlKSB7XG4gIHZhciB0ID0ge307XG5cbiAgZm9yICh2YXIgcCBpbiBzKSB7XG4gICAgaWYgKE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChzLCBwKSAmJiBlLmluZGV4T2YocCkgPCAwKSB0W3BdID0gc1twXTtcbiAgfVxuXG4gIGlmIChzICE9IG51bGwgJiYgdHlwZW9mIE9iamVjdC5nZXRPd25Qcm9wZXJ0eVN5bWJvbHMgPT09IFwiZnVuY3Rpb25cIikgZm9yICh2YXIgaSA9IDAsIHAgPSBPYmplY3QuZ2V0T3duUHJvcGVydHlTeW1ib2xzKHMpOyBpIDwgcC5sZW5ndGg7IGkrKykge1xuICAgIGlmIChlLmluZGV4T2YocFtpXSkgPCAwICYmIE9iamVjdC5wcm90b3R5cGUucHJvcGVydHlJc0VudW1lcmFibGUuY2FsbChzLCBwW2ldKSkgdFtwW2ldXSA9IHNbcFtpXV07XG4gIH1cbiAgcmV0dXJuIHQ7XG59O1xuXG52YXIgZGVmYXVsdFJvb3RQcmVmaXhDbHMgPSAnYW50JztcblxuZnVuY3Rpb24gZ2V0Um9vdFByZWZpeENscygpIHtcbiAgcmV0dXJuIGRlZmF1bHRSb290UHJlZml4Q2xzO1xufVxuXG5mdW5jdGlvbiBjb25maXJtKGNvbmZpZykge1xuICB2YXIgZGl2ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7XG4gIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQoZGl2KTsgLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIEB0eXBlc2NyaXB0LWVzbGludC9uby11c2UtYmVmb3JlLWRlZmluZVxuXG4gIHZhciBjdXJyZW50Q29uZmlnID0gKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKCgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7fSwgY29uZmlnKSwge1xuICAgIGNsb3NlOiBjbG9zZSxcbiAgICB2aXNpYmxlOiB0cnVlXG4gIH0pO1xuXG4gIGZ1bmN0aW9uIGRlc3Ryb3koKSB7XG4gICAgdmFyIHVubW91bnRSZXN1bHQgPSBSZWFjdERPTS51bm1vdW50Q29tcG9uZW50QXROb2RlKGRpdik7XG5cbiAgICBpZiAodW5tb3VudFJlc3VsdCAmJiBkaXYucGFyZW50Tm9kZSkge1xuICAgICAgZGl2LnBhcmVudE5vZGUucmVtb3ZlQ2hpbGQoZGl2KTtcbiAgICB9XG5cbiAgICBmb3IgKHZhciBfbGVuID0gYXJndW1lbnRzLmxlbmd0aCwgYXJncyA9IG5ldyBBcnJheShfbGVuKSwgX2tleSA9IDA7IF9rZXkgPCBfbGVuOyBfa2V5KyspIHtcbiAgICAgIGFyZ3NbX2tleV0gPSBhcmd1bWVudHNbX2tleV07XG4gICAgfVxuXG4gICAgdmFyIHRyaWdnZXJDYW5jZWwgPSBhcmdzLnNvbWUoZnVuY3Rpb24gKHBhcmFtKSB7XG4gICAgICByZXR1cm4gcGFyYW0gJiYgcGFyYW0udHJpZ2dlckNhbmNlbDtcbiAgICB9KTtcblxuICAgIGlmIChjb25maWcub25DYW5jZWwgJiYgdHJpZ2dlckNhbmNlbCkge1xuICAgICAgY29uZmlnLm9uQ2FuY2VsLmFwcGx5KGNvbmZpZywgYXJncyk7XG4gICAgfVxuXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBfTW9kYWwuZGVzdHJveUZucy5sZW5ndGg7IGkrKykge1xuICAgICAgdmFyIGZuID0gX01vZGFsLmRlc3Ryb3lGbnNbaV07IC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBAdHlwZXNjcmlwdC1lc2xpbnQvbm8tdXNlLWJlZm9yZS1kZWZpbmVcblxuICAgICAgaWYgKGZuID09PSBjbG9zZSkge1xuICAgICAgICBfTW9kYWwuZGVzdHJveUZucy5zcGxpY2UoaSwgMSk7XG5cbiAgICAgICAgYnJlYWs7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZnVuY3Rpb24gcmVuZGVyKF9hKSB7XG4gICAgdmFyIG9rVGV4dCA9IF9hLm9rVGV4dCxcbiAgICAgICAgY2FuY2VsVGV4dCA9IF9hLmNhbmNlbFRleHQsXG4gICAgICAgIHByZWZpeENscyA9IF9hLnByZWZpeENscyxcbiAgICAgICAgcHJvcHMgPSBfX3Jlc3QoX2EsIFtcIm9rVGV4dFwiLCBcImNhbmNlbFRleHRcIiwgXCJwcmVmaXhDbHNcIl0pO1xuICAgIC8qKlxuICAgICAqIGh0dHBzOi8vZ2l0aHViLmNvbS9hbnQtZGVzaWduL2FudC1kZXNpZ24vaXNzdWVzLzIzNjIzXG4gICAgICpcbiAgICAgKiBTeW5jIHJlbmRlciBibG9ja3MgUmVhY3QgZXZlbnQuIExldCdzIG1ha2UgdGhpcyBhc3luYy5cbiAgICAgKi9cblxuXG4gICAgc2V0VGltZW91dChmdW5jdGlvbiAoKSB7XG4gICAgICB2YXIgcnVudGltZUxvY2FsZSA9ICgwLCBfbG9jYWxlLmdldENvbmZpcm1Mb2NhbGUpKCk7XG4gICAgICBSZWFjdERPTS5yZW5kZXIoIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9Db25maXJtRGlhbG9nW1wiZGVmYXVsdFwiXSwgKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHt9LCBwcm9wcywge1xuICAgICAgICBwcmVmaXhDbHM6IHByZWZpeENscyB8fCBcIlwiLmNvbmNhdChnZXRSb290UHJlZml4Q2xzKCksIFwiLW1vZGFsXCIpLFxuICAgICAgICByb290UHJlZml4Q2xzOiBnZXRSb290UHJlZml4Q2xzKCksXG4gICAgICAgIG9rVGV4dDogb2tUZXh0IHx8IChwcm9wcy5va0NhbmNlbCA/IHJ1bnRpbWVMb2NhbGUub2tUZXh0IDogcnVudGltZUxvY2FsZS5qdXN0T2tUZXh0KSxcbiAgICAgICAgY2FuY2VsVGV4dDogY2FuY2VsVGV4dCB8fCBydW50aW1lTG9jYWxlLmNhbmNlbFRleHRcbiAgICAgIH0pKSwgZGl2KTtcbiAgICB9KTtcbiAgfVxuXG4gIGZ1bmN0aW9uIGNsb3NlKCkge1xuICAgIHZhciBfdGhpcyA9IHRoaXM7XG5cbiAgICBmb3IgKHZhciBfbGVuMiA9IGFyZ3VtZW50cy5sZW5ndGgsIGFyZ3MgPSBuZXcgQXJyYXkoX2xlbjIpLCBfa2V5MiA9IDA7IF9rZXkyIDwgX2xlbjI7IF9rZXkyKyspIHtcbiAgICAgIGFyZ3NbX2tleTJdID0gYXJndW1lbnRzW19rZXkyXTtcbiAgICB9XG5cbiAgICBjdXJyZW50Q29uZmlnID0gKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKCgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7fSwgY3VycmVudENvbmZpZyksIHtcbiAgICAgIHZpc2libGU6IGZhbHNlLFxuICAgICAgYWZ0ZXJDbG9zZTogZnVuY3Rpb24gYWZ0ZXJDbG9zZSgpIHtcbiAgICAgICAgaWYgKHR5cGVvZiBjb25maWcuYWZ0ZXJDbG9zZSA9PT0gJ2Z1bmN0aW9uJykge1xuICAgICAgICAgIGNvbmZpZy5hZnRlckNsb3NlKCk7XG4gICAgICAgIH1cblxuICAgICAgICBkZXN0cm95LmFwcGx5KF90aGlzLCBhcmdzKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICByZW5kZXIoY3VycmVudENvbmZpZyk7XG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGUoY29uZmlnVXBkYXRlKSB7XG4gICAgaWYgKHR5cGVvZiBjb25maWdVcGRhdGUgPT09ICdmdW5jdGlvbicpIHtcbiAgICAgIGN1cnJlbnRDb25maWcgPSBjb25maWdVcGRhdGUoY3VycmVudENvbmZpZyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGN1cnJlbnRDb25maWcgPSAoMCwgX2V4dGVuZHMyW1wiZGVmYXVsdFwiXSkoKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHt9LCBjdXJyZW50Q29uZmlnKSwgY29uZmlnVXBkYXRlKTtcbiAgICB9XG5cbiAgICByZW5kZXIoY3VycmVudENvbmZpZyk7XG4gIH1cblxuICByZW5kZXIoY3VycmVudENvbmZpZyk7XG5cbiAgX01vZGFsLmRlc3Ryb3lGbnMucHVzaChjbG9zZSk7XG5cbiAgcmV0dXJuIHtcbiAgICBkZXN0cm95OiBjbG9zZSxcbiAgICB1cGRhdGU6IHVwZGF0ZVxuICB9O1xufVxuXG5mdW5jdGlvbiB3aXRoV2Fybihwcm9wcykge1xuICByZXR1cm4gKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKCgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7XG4gICAgaWNvbjogLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0V4Y2xhbWF0aW9uQ2lyY2xlT3V0bGluZWRbXCJkZWZhdWx0XCJdLCBudWxsKSxcbiAgICBva0NhbmNlbDogZmFsc2VcbiAgfSwgcHJvcHMpLCB7XG4gICAgdHlwZTogJ3dhcm5pbmcnXG4gIH0pO1xufVxuXG5mdW5jdGlvbiB3aXRoSW5mbyhwcm9wcykge1xuICByZXR1cm4gKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKCgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7XG4gICAgaWNvbjogLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0luZm9DaXJjbGVPdXRsaW5lZFtcImRlZmF1bHRcIl0sIG51bGwpLFxuICAgIG9rQ2FuY2VsOiBmYWxzZVxuICB9LCBwcm9wcyksIHtcbiAgICB0eXBlOiAnaW5mbydcbiAgfSk7XG59XG5cbmZ1bmN0aW9uIHdpdGhTdWNjZXNzKHByb3BzKSB7XG4gIHJldHVybiAoMCwgX2V4dGVuZHMyW1wiZGVmYXVsdFwiXSkoKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHtcbiAgICBpY29uOiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfQ2hlY2tDaXJjbGVPdXRsaW5lZFtcImRlZmF1bHRcIl0sIG51bGwpLFxuICAgIG9rQ2FuY2VsOiBmYWxzZVxuICB9LCBwcm9wcyksIHtcbiAgICB0eXBlOiAnc3VjY2VzcydcbiAgfSk7XG59XG5cbmZ1bmN0aW9uIHdpdGhFcnJvcihwcm9wcykge1xuICByZXR1cm4gKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKCgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7XG4gICAgaWNvbjogLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoX0Nsb3NlQ2lyY2xlT3V0bGluZWRbXCJkZWZhdWx0XCJdLCBudWxsKSxcbiAgICBva0NhbmNlbDogZmFsc2VcbiAgfSwgcHJvcHMpLCB7XG4gICAgdHlwZTogJ2Vycm9yJ1xuICB9KTtcbn1cblxuZnVuY3Rpb24gd2l0aENvbmZpcm0ocHJvcHMpIHtcbiAgcmV0dXJuICgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSgoMCwgX2V4dGVuZHMyW1wiZGVmYXVsdFwiXSkoe1xuICAgIGljb246IC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9FeGNsYW1hdGlvbkNpcmNsZU91dGxpbmVkW1wiZGVmYXVsdFwiXSwgbnVsbCksXG4gICAgb2tDYW5jZWw6IHRydWVcbiAgfSwgcHJvcHMpLCB7XG4gICAgdHlwZTogJ2NvbmZpcm0nXG4gIH0pO1xufVxuXG5mdW5jdGlvbiBnbG9iYWxDb25maWcoX3JlZikge1xuICB2YXIgcm9vdFByZWZpeENscyA9IF9yZWYucm9vdFByZWZpeENscztcblxuICBpZiAocm9vdFByZWZpeENscykge1xuICAgIGRlZmF1bHRSb290UHJlZml4Q2xzID0gcm9vdFByZWZpeENscztcbiAgfVxufSIsIlwidXNlIHN0cmljdFwiO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkXCIpO1xuXG52YXIgX2ludGVyb3BSZXF1aXJlRGVmYXVsdCA9IHJlcXVpcmUoXCJAYmFiZWwvcnVudGltZS9oZWxwZXJzL2ludGVyb3BSZXF1aXJlRGVmYXVsdFwiKTtcblxuT2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsIFwiX19lc01vZHVsZVwiLCB7XG4gIHZhbHVlOiB0cnVlXG59KTtcbmV4cG9ydHNbXCJkZWZhdWx0XCJdID0gdm9pZCAwO1xuXG52YXIgX2V4dGVuZHMyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9leHRlbmRzXCIpKTtcblxudmFyIF9zbGljZWRUb0FycmF5MiA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvc2xpY2VkVG9BcnJheVwiKSk7XG5cbnZhciBSZWFjdCA9IF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkKHJlcXVpcmUoXCJyZWFjdFwiKSk7XG5cbnZhciBfQ29uZmlybURpYWxvZyA9IF9pbnRlcm9wUmVxdWlyZURlZmF1bHQocmVxdWlyZShcIi4uL0NvbmZpcm1EaWFsb2dcIikpO1xuXG52YXIgX2RlZmF1bHQyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi4vLi4vbG9jYWxlL2RlZmF1bHRcIikpO1xuXG52YXIgX0xvY2FsZVJlY2VpdmVyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi4vLi4vbG9jYWxlLXByb3ZpZGVyL0xvY2FsZVJlY2VpdmVyXCIpKTtcblxudmFyIF9jb25maWdQcm92aWRlciA9IHJlcXVpcmUoXCIuLi8uLi9jb25maWctcHJvdmlkZXJcIik7XG5cbnZhciBIb29rTW9kYWwgPSBmdW5jdGlvbiBIb29rTW9kYWwoX3JlZiwgcmVmKSB7XG4gIHZhciBhZnRlckNsb3NlID0gX3JlZi5hZnRlckNsb3NlLFxuICAgICAgY29uZmlnID0gX3JlZi5jb25maWc7XG5cbiAgdmFyIF9SZWFjdCR1c2VTdGF0ZSA9IFJlYWN0LnVzZVN0YXRlKHRydWUpLFxuICAgICAgX1JlYWN0JHVzZVN0YXRlMiA9ICgwLCBfc2xpY2VkVG9BcnJheTJbXCJkZWZhdWx0XCJdKShfUmVhY3QkdXNlU3RhdGUsIDIpLFxuICAgICAgdmlzaWJsZSA9IF9SZWFjdCR1c2VTdGF0ZTJbMF0sXG4gICAgICBzZXRWaXNpYmxlID0gX1JlYWN0JHVzZVN0YXRlMlsxXTtcblxuICB2YXIgX1JlYWN0JHVzZVN0YXRlMyA9IFJlYWN0LnVzZVN0YXRlKGNvbmZpZyksXG4gICAgICBfUmVhY3QkdXNlU3RhdGU0ID0gKDAsIF9zbGljZWRUb0FycmF5MltcImRlZmF1bHRcIl0pKF9SZWFjdCR1c2VTdGF0ZTMsIDIpLFxuICAgICAgaW5uZXJDb25maWcgPSBfUmVhY3QkdXNlU3RhdGU0WzBdLFxuICAgICAgc2V0SW5uZXJDb25maWcgPSBfUmVhY3QkdXNlU3RhdGU0WzFdO1xuXG4gIHZhciBfUmVhY3QkdXNlQ29udGV4dCA9IFJlYWN0LnVzZUNvbnRleHQoX2NvbmZpZ1Byb3ZpZGVyLkNvbmZpZ0NvbnRleHQpLFxuICAgICAgZGlyZWN0aW9uID0gX1JlYWN0JHVzZUNvbnRleHQuZGlyZWN0aW9uLFxuICAgICAgZ2V0UHJlZml4Q2xzID0gX1JlYWN0JHVzZUNvbnRleHQuZ2V0UHJlZml4Q2xzO1xuXG4gIHZhciBwcmVmaXhDbHMgPSBnZXRQcmVmaXhDbHMoJ21vZGFsJyk7XG4gIHZhciByb290UHJlZml4Q2xzID0gZ2V0UHJlZml4Q2xzKCk7XG5cbiAgZnVuY3Rpb24gY2xvc2UoKSB7XG4gICAgc2V0VmlzaWJsZShmYWxzZSk7XG5cbiAgICBmb3IgKHZhciBfbGVuID0gYXJndW1lbnRzLmxlbmd0aCwgYXJncyA9IG5ldyBBcnJheShfbGVuKSwgX2tleSA9IDA7IF9rZXkgPCBfbGVuOyBfa2V5KyspIHtcbiAgICAgIGFyZ3NbX2tleV0gPSBhcmd1bWVudHNbX2tleV07XG4gICAgfVxuXG4gICAgdmFyIHRyaWdnZXJDYW5jZWwgPSBhcmdzLnNvbWUoZnVuY3Rpb24gKHBhcmFtKSB7XG4gICAgICByZXR1cm4gcGFyYW0gJiYgcGFyYW0udHJpZ2dlckNhbmNlbDtcbiAgICB9KTtcblxuICAgIGlmIChpbm5lckNvbmZpZy5vbkNhbmNlbCAmJiB0cmlnZ2VyQ2FuY2VsKSB7XG4gICAgICBpbm5lckNvbmZpZy5vbkNhbmNlbCgpO1xuICAgIH1cbiAgfVxuXG4gIFJlYWN0LnVzZUltcGVyYXRpdmVIYW5kbGUocmVmLCBmdW5jdGlvbiAoKSB7XG4gICAgcmV0dXJuIHtcbiAgICAgIGRlc3Ryb3k6IGNsb3NlLFxuICAgICAgdXBkYXRlOiBmdW5jdGlvbiB1cGRhdGUobmV3Q29uZmlnKSB7XG4gICAgICAgIHNldElubmVyQ29uZmlnKGZ1bmN0aW9uIChvcmlnaW5Db25maWcpIHtcbiAgICAgICAgICByZXR1cm4gKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKCgwLCBfZXh0ZW5kczJbXCJkZWZhdWx0XCJdKSh7fSwgb3JpZ2luQ29uZmlnKSwgbmV3Q29uZmlnKTtcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgfTtcbiAgfSk7XG4gIHJldHVybiAvKiNfX1BVUkVfXyovUmVhY3QuY3JlYXRlRWxlbWVudChfTG9jYWxlUmVjZWl2ZXJbXCJkZWZhdWx0XCJdLCB7XG4gICAgY29tcG9uZW50TmFtZTogXCJNb2RhbFwiLFxuICAgIGRlZmF1bHRMb2NhbGU6IF9kZWZhdWx0MltcImRlZmF1bHRcIl0uTW9kYWxcbiAgfSwgZnVuY3Rpb24gKG1vZGFsTG9jYWxlKSB7XG4gICAgcmV0dXJuIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9Db25maXJtRGlhbG9nW1wiZGVmYXVsdFwiXSwgKDAsIF9leHRlbmRzMltcImRlZmF1bHRcIl0pKHtcbiAgICAgIHByZWZpeENsczogcHJlZml4Q2xzLFxuICAgICAgcm9vdFByZWZpeENsczogcm9vdFByZWZpeENsc1xuICAgIH0sIGlubmVyQ29uZmlnLCB7XG4gICAgICBjbG9zZTogY2xvc2UsXG4gICAgICB2aXNpYmxlOiB2aXNpYmxlLFxuICAgICAgYWZ0ZXJDbG9zZTogYWZ0ZXJDbG9zZSxcbiAgICAgIG9rVGV4dDogaW5uZXJDb25maWcub2tUZXh0IHx8IChpbm5lckNvbmZpZy5va0NhbmNlbCA/IG1vZGFsTG9jYWxlLm9rVGV4dCA6IG1vZGFsTG9jYWxlLmp1c3RPa1RleHQpLFxuICAgICAgZGlyZWN0aW9uOiBkaXJlY3Rpb24sXG4gICAgICBjYW5jZWxUZXh0OiBpbm5lckNvbmZpZy5jYW5jZWxUZXh0IHx8IG1vZGFsTG9jYWxlLmNhbmNlbFRleHRcbiAgICB9KSk7XG4gIH0pO1xufTtcblxudmFyIF9kZWZhdWx0ID0gLyojX19QVVJFX18qL1JlYWN0LmZvcndhcmRSZWYoSG9va01vZGFsKTtcblxuZXhwb3J0c1tcImRlZmF1bHRcIl0gPSBfZGVmYXVsdDsiLCJcInVzZSBzdHJpY3RcIjtcblxudmFyIF9pbnRlcm9wUmVxdWlyZVdpbGRjYXJkID0gcmVxdWlyZShcIkBiYWJlbC9ydW50aW1lL2hlbHBlcnMvaW50ZXJvcFJlcXVpcmVXaWxkY2FyZFwiKTtcblxudmFyIF9pbnRlcm9wUmVxdWlyZURlZmF1bHQgPSByZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9pbnRlcm9wUmVxdWlyZURlZmF1bHRcIik7XG5cbk9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBcIl9fZXNNb2R1bGVcIiwge1xuICB2YWx1ZTogdHJ1ZVxufSk7XG5leHBvcnRzW1wiZGVmYXVsdFwiXSA9IHVzZU1vZGFsO1xuXG52YXIgX3NsaWNlZFRvQXJyYXkyID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiQGJhYmVsL3J1bnRpbWUvaGVscGVycy9zbGljZWRUb0FycmF5XCIpKTtcblxudmFyIFJlYWN0ID0gX2ludGVyb3BSZXF1aXJlV2lsZGNhcmQocmVxdWlyZShcInJlYWN0XCIpKTtcblxudmFyIF91c2VQYXRjaEVsZW1lbnQzID0gX2ludGVyb3BSZXF1aXJlRGVmYXVsdChyZXF1aXJlKFwiLi4vLi4vX3V0aWwvaG9va3MvdXNlUGF0Y2hFbGVtZW50XCIpKTtcblxudmFyIF9Ib29rTW9kYWwgPSBfaW50ZXJvcFJlcXVpcmVEZWZhdWx0KHJlcXVpcmUoXCIuL0hvb2tNb2RhbFwiKSk7XG5cbnZhciBfY29uZmlybSA9IHJlcXVpcmUoXCIuLi9jb25maXJtXCIpO1xuXG52YXIgdXVpZCA9IDA7XG52YXIgRWxlbWVudHNIb2xkZXIgPSAvKiNfX1BVUkVfXyovUmVhY3QubWVtbyggLyojX19QVVJFX18qL1JlYWN0LmZvcndhcmRSZWYoZnVuY3Rpb24gKF9wcm9wcywgcmVmKSB7XG4gIHZhciBfdXNlUGF0Y2hFbGVtZW50ID0gKDAsIF91c2VQYXRjaEVsZW1lbnQzW1wiZGVmYXVsdFwiXSkoKSxcbiAgICAgIF91c2VQYXRjaEVsZW1lbnQyID0gKDAsIF9zbGljZWRUb0FycmF5MltcImRlZmF1bHRcIl0pKF91c2VQYXRjaEVsZW1lbnQsIDIpLFxuICAgICAgZWxlbWVudHMgPSBfdXNlUGF0Y2hFbGVtZW50MlswXSxcbiAgICAgIHBhdGNoRWxlbWVudCA9IF91c2VQYXRjaEVsZW1lbnQyWzFdO1xuXG4gIFJlYWN0LnVzZUltcGVyYXRpdmVIYW5kbGUocmVmLCBmdW5jdGlvbiAoKSB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHBhdGNoRWxlbWVudDogcGF0Y2hFbGVtZW50XG4gICAgfTtcbiAgfSwgW10pO1xuICByZXR1cm4gLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZUVsZW1lbnQoUmVhY3QuRnJhZ21lbnQsIG51bGwsIGVsZW1lbnRzKTtcbn0pKTtcblxuZnVuY3Rpb24gdXNlTW9kYWwoKSB7XG4gIHZhciBob2xkZXJSZWYgPSBSZWFjdC51c2VSZWYobnVsbCk7XG4gIHZhciBnZXRDb25maXJtRnVuYyA9IFJlYWN0LnVzZUNhbGxiYWNrKGZ1bmN0aW9uICh3aXRoRnVuYykge1xuICAgIHJldHVybiBmdW5jdGlvbiBob29rQ29uZmlybShjb25maWcpIHtcbiAgICAgIHZhciBfYTtcblxuICAgICAgdXVpZCArPSAxO1xuICAgICAgdmFyIG1vZGFsUmVmID0gLyojX19QVVJFX18qL1JlYWN0LmNyZWF0ZVJlZigpO1xuICAgICAgdmFyIGNsb3NlRnVuYztcbiAgICAgIHZhciBtb2RhbCA9IC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KF9Ib29rTW9kYWxbXCJkZWZhdWx0XCJdLCB7XG4gICAgICAgIGtleTogXCJtb2RhbC1cIi5jb25jYXQodXVpZCksXG4gICAgICAgIGNvbmZpZzogd2l0aEZ1bmMoY29uZmlnKSxcbiAgICAgICAgcmVmOiBtb2RhbFJlZixcbiAgICAgICAgYWZ0ZXJDbG9zZTogZnVuY3Rpb24gYWZ0ZXJDbG9zZSgpIHtcbiAgICAgICAgICBjbG9zZUZ1bmMoKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgICBjbG9zZUZ1bmMgPSAoX2EgPSBob2xkZXJSZWYuY3VycmVudCkgPT09IG51bGwgfHwgX2EgPT09IHZvaWQgMCA/IHZvaWQgMCA6IF9hLnBhdGNoRWxlbWVudChtb2RhbCk7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBkZXN0cm95OiBmdW5jdGlvbiBkZXN0cm95KCkge1xuICAgICAgICAgIGlmIChtb2RhbFJlZi5jdXJyZW50KSB7XG4gICAgICAgICAgICBtb2RhbFJlZi5jdXJyZW50LmRlc3Ryb3koKTtcbiAgICAgICAgICB9XG4gICAgICAgIH0sXG4gICAgICAgIHVwZGF0ZTogZnVuY3Rpb24gdXBkYXRlKG5ld0NvbmZpZykge1xuICAgICAgICAgIGlmIChtb2RhbFJlZi5jdXJyZW50KSB7XG4gICAgICAgICAgICBtb2RhbFJlZi5jdXJyZW50LnVwZGF0ZShuZXdDb25maWcpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfTtcbiAgICB9O1xuICB9LCBbXSk7XG4gIHZhciBmbnMgPSBSZWFjdC51c2VNZW1vKGZ1bmN0aW9uICgpIHtcbiAgICByZXR1cm4ge1xuICAgICAgaW5mbzogZ2V0Q29uZmlybUZ1bmMoX2NvbmZpcm0ud2l0aEluZm8pLFxuICAgICAgc3VjY2VzczogZ2V0Q29uZmlybUZ1bmMoX2NvbmZpcm0ud2l0aFN1Y2Nlc3MpLFxuICAgICAgZXJyb3I6IGdldENvbmZpcm1GdW5jKF9jb25maXJtLndpdGhFcnJvciksXG4gICAgICB3YXJuaW5nOiBnZXRDb25maXJtRnVuYyhfY29uZmlybS53aXRoV2FybiksXG4gICAgICBjb25maXJtOiBnZXRDb25maXJtRnVuYyhfY29uZmlybS53aXRoQ29uZmlybSlcbiAgICB9O1xuICB9LCBbXSk7IC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSByZWFjdC9qc3gta2V5XG5cbiAgcmV0dXJuIFtmbnMsIC8qI19fUFVSRV9fKi9SZWFjdC5jcmVhdGVFbGVtZW50KEVsZW1lbnRzSG9sZGVyLCB7XG4gICAgcmVmOiBob2xkZXJSZWZcbiAgfSldO1xufSJdLCJzb3VyY2VSb290IjoiIn0=